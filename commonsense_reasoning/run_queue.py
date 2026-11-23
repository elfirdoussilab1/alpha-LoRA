"""Minimal resumable multi-GPU job queue.

Job file: one job per line, tab separated:  RUN_ID <TAB> OUT_DIR <TAB> GPU_COST <TAB> COMMAND
  * a job is DONE if OUT_DIR/RUN_ID.json exists -> re-running the queue skips it (resumable);
  * GPU_COST in (0, 1]: fraction of one GPU (0.25 => 4 small jobs share a GPU, 1 => exclusive);
  * stdout/stderr -> <log_dir>/RUN_ID.log ; queue state -> <log_dir>/_queue_status.json (read by monitor.sh)
"""
import argparse
import json
import os
import subprocess
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--poll", type=float, default=5.0)
    args = ap.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    gpus = [g for g in args.gpus.split(",") if g != ""]
    load = {g: 0.0 for g in gpus}

    jobs = []
    for line in open(args.jobs):
        line = line.rstrip("\n")
        if not line.strip() or line.startswith("#"):
            continue
        rid, out_dir, cost, cmd = line.split("\t", 3)
        jobs.append(dict(rid=rid, out=out_dir, cost=float(cost), cmd=cmd, tries=0))
    is_done = lambda j: os.path.exists(os.path.join(j["out"], f"{j['rid']}.json"))
    pending = [j for j in jobs if not is_done(j)]
    n_skip = len(jobs) - len(pending)
    pending.sort(key=lambda j: -j["cost"])   # big jobs first: better packing
    running, failed, n_done = [], [], 0
    print(f"[queue] {len(jobs)} jobs, {n_skip} already done, {len(pending)} to run on GPUs {gpus}", flush=True)

    def write_status():
        st = {"total": len(jobs), "done": n_skip + n_done, "running": [(j["rid"], j["gpu"]) for j in running],
              "pending": len(pending), "failed": [j["rid"] for j in failed], "time": time.strftime("%H:%M:%S")}
        json.dump(st, open(os.path.join(args.log_dir, "_queue_status.json"), "w"), indent=1)

    while pending or running:
        # reap
        for j in list(running):
            rc = j["proc"].poll()
            if rc is None:
                continue
            running.remove(j); load[j["gpu"]] -= j["cost"]; j["fh"].close()
            if rc == 0 and is_done(j):
                n_done += 1
                print(f"[queue] done   {j['rid']} ({(time.time()-j['t0'])/60:.1f} min)", flush=True)
            elif j["tries"] <= args.retries:
                print(f"[queue] retry  {j['rid']} (rc={rc})", flush=True); pending.append(j)
            else:
                failed.append(j); print(f"[queue] FAILED {j['rid']} (rc={rc}) see {args.log_dir}/{j['rid']}.log", flush=True)
        # launch
        for j in list(pending):
            g = min(gpus, key=lambda x: load[x])
            if load[g] + j["cost"] > 1.0 + 1e-6:
                continue
            pending.remove(j)
            j["tries"] += 1; j["gpu"] = g; j["t0"] = time.time()
            j["fh"] = open(os.path.join(args.log_dir, f"{j['rid']}.log"), "a")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, TOKENIZERS_PARALLELISM="false")
            j["proc"] = subprocess.Popen(j["cmd"], shell=True, stdout=j["fh"], stderr=subprocess.STDOUT, env=env)
            load[g] += j["cost"]; running.append(j)
            print(f"[queue] start  {j['rid']} on GPU {g}", flush=True)
        write_status()
        time.sleep(args.poll)
    write_status()
    print(f"[queue] finished: {n_skip + n_done}/{len(jobs)} done, {len(failed)} failed", flush=True)
    for j in failed:
        print(f"    failed: {j['rid']}")


if __name__ == "__main__":
    main()
