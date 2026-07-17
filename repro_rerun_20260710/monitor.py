#!/usr/bin/env python3
"""Monitor + QA the rerun: flatness, cross-state gradient (pristine vs recheck)."""
import json, glob, os
BASE="benchmark/tests/repro_rerun_20260710"

def load(fp):
    d=json.load(open(fp))[0]
    its=[x for x in d["iterations"] if not x["is_warmup"]]
    tt=[x["ttft_ms"] for x in its]
    return dict(mean=d["ttft_mean_ms"], std=d["ttft_std_ms"], tpot=d["tpot_mean_ms"],
                tps=d["throughput_mean_tps"], dtps=d["decode_throughput_mean_tps"],
                e2e=d["e2e_mean_ms"], n=len(tt), first=tt[0], last=tt[-1],
                md5=[x["result_md5"] for x in its])

print("========= GPU =========")
for size in ["0.8B-INT4","2B-INT4","4B-INT4"]:
    print(f"\n-- {size} --")
    pj=f"{BASE}/gpu_pa/{size}/pristine.json"; rj=f"{BASE}/gpu_pa/{size}/pristine_recheck.json"
    for lbl in ["pristine","opt3","opt3_opt5","opt3_opt5_opt6","pristine_recheck"]:
        fp=f"{BASE}/gpu_pa/{size}/{lbl}.json"
        if not os.path.exists(fp): print(f"  {lbl:<18} ..pending.."); continue
        r=load(fp)
        flag=""
        if r["std"]>8: flag+=" HI-STD!"
        if abs(r["last"]-r["first"])>15: flag+=" RAMP!"
        print(f"  {lbl:<18} mean={r['mean']:7.2f} std={r['std']:5.2f} {r['first']:.0f}->{r['last']:.0f} TPOT={r['tpot']:5.2f} TPS={r['tps']:6.2f} E2E={r['e2e']:8.1f}{flag}")
    if os.path.exists(pj) and os.path.exists(rj):
        p=load(pj); rc=load(rj); d=rc["mean"]-p["mean"]
        g="OK" if abs(d)<=max(6,2*(p['std']+rc['std'])) else "GRADIENT WARNING"
        print(f"  >>> gradient check: pristine={p['mean']:.2f} recheck={rc['mean']:.2f} Δ={d:+.2f}  [{g}]")

print("\n========= NPU =========")
for m in ["Qwen3-0.6B-INT4","Qwen3-1.7B-INT4","Qwen3-4B-INT4"]:
    print(f"\n-- {m} --")
    for lbl in ["base","turbo","best_perf","best_perf_turbo","base_recheck"]:
        fp=f"{BASE}/npu/{m}_{lbl}.json"
        if not os.path.exists(fp):
            if lbl!="base_recheck" or m=="Qwen3-4B-INT4": print(f"  {lbl:<16} ..pending..")
            continue
        r=load(fp)
        flag=" RAMP!" if abs(r["last"]-r["first"])>15 else ""
        print(f"  {lbl:<16} mean={r['mean']:7.2f} std={r['std']:5.2f} {r['first']:.0f}->{r['last']:.0f} TPOT={r['tpot']:5.2f} TPS={r['tps']:6.2f} E2E={r['e2e']:8.1f}{flag}")
