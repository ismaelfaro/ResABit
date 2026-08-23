#!/usr/bin/env python
"""Recompute every headline statistic in the paper from raw ledger rows.

This is the provenance artifact for the derived statistics quoted in
paper/preprint.md (shares, paired SE, SE ratio): the audit of 2026-08
found the paired SE was emitted by no repo script, so this one exists.

Reads ONLY raw ledger rows from results/*.jsonl; shares/interactions/SEs are
recomputed from loss_nats and log(151936), never taken from derived ledger
fields (ledger 'headroom' fields are cross-checked but not used).

Run:  cd /Users/ismaelsertage/code/ResABit && .venv/bin/python <this file>
"""
import json
import math
import statistics
import sys

import os
REPO = os.path.dirname(os.path.abspath(__file__))
VOCAB = 151936
FLOOR = math.log(VOCAB)


def rows(path):
    with open(f"{REPO}/results/{path}") as f:
        return [json.loads(line) for line in f if line.strip()]


def key(r, i):
    return (f"line{i+1}:cell={r.get('cell')},seed={r.get('seed')},"
            f"stage={r.get('stage')},steps={r['train_config']['steps']},"
            f"commit={r.get('commit')}")


def check(label, computed, claimed, tol, loc):
    ok = abs(computed - claimed) < tol
    flag = "OK " if ok else "MISMATCH"
    print(f"  [{flag}] {label}: computed={computed:.6f} claimed={claimed} ({loc})")
    return ok


print("=" * 78)
print("0. Uniform floor")
print("=" * 78)
print(f"  log(151936) = {FLOOR:.10f}  -> rounds to {FLOOR:.4f}")
check("floor", FLOOR, 11.9312, 0.00005, "preprint.md:543,570")

# ---------------------------------------------------------------- grid ledger
grid = rows("grid_ledger.jsonl")
print(f"\ngrid_ledger.jsonl: {len(grid)} rows")
for i, r in enumerate(grid):
    uf = r.get("uniform_floor")
    if uf is not None:
        assert abs(uf - FLOOR) < 1e-9, f"row {i}: ledger floor {uf} != log(151936)"
    hr = r.get("headroom")
    if hr is not None:
        assert abs(hr - (FLOOR - r["loss_nats"])) < 1e-6, f"row {i}: ledger headroom inconsistent"

full = [(i, r) for i, r in enumerate(grid) if r.get("stage") == "full" and r.get("status") == "ok"]
smoke = [(i, r) for i, r in enumerate(grid) if r.get("stage") == "smoke"]
print(f"  full-stage rows: {len(full)}, smoke rows excluded: {len(smoke)}")

by_budget = {}
for i, r in full:
    by_budget.setdefault(r["train_config"]["steps"], []).append((i, r))
for steps, rs in sorted(by_budget.items()):
    tokens = steps * 2 * 4 * 512
    print(f"  budget steps={steps} ({tokens/1e6:.3f}M tokens): {len(rs)} rows, "
          f"seeds={sorted(r['seed'] for _, r in rs)}")

BASE = 300
base = {(r["cell"], r["seed"]): (i, r) for i, r in by_budget[BASE]}
assert len(base) == 12, "expected 12 base-budget rows (4 cells x 3 seeds)"

print("\n" + "=" * 78)
print("1. Per-cell mean eval loss at base budget (steps=300, seeds 0,1,2)")
print("=" * 78)
claimed_means = {"fp32_diff": 3.8702, "ternary_diff": 5.9408,
                 "fp32_ar": 2.6909, "ternary_ar": 4.0350}
cell_means = {}
for cell in ["fp32_diff", "ternary_diff", "fp32_ar", "ternary_ar"]:
    losses = [base[(cell, s)][1]["loss_nats"] for s in (0, 1, 2)]
    m = statistics.fmean(losses)
    cell_means[cell] = m
    print(f"  {cell}: per-seed {['%.4f' % x for x in losses]} mean={m:.6f} "
          f"headroom={FLOOR - m:.6f}")
    check(f"{cell} mean loss", m, claimed_means[cell], 0.0005, "preprint.md:582-585")
    rowkeys = [key(*reversed(base[(cell, s)])) for s in (0, 1, 2)]
    for rk in rowkeys:
        print(f"      row {rk}")

claimed_headrooms = {"fp32_diff": 8.0610, "ternary_diff": 5.9904,
                     "fp32_ar": 9.2403, "ternary_ar": 7.8962}
for cell, ch in claimed_headrooms.items():
    check(f"{cell} mean headroom", FLOOR - cell_means[cell], ch, 0.0005,
          "preprint.md:582-585")

print("\n" + "=" * 78)
print("2. Per-seed share of headroom destroyed (paired)")
print("=" * 78)


def share(seed, fp32_cell, tern_cell, budget=BASE):
    d = {(r["cell"], r["seed"]): r for _, r in by_budget[budget]}
    hf = FLOOR - d[(fp32_cell, seed)]["loss_nats"]
    ht = FLOOR - d[(tern_cell, seed)]["loss_nats"]
    return (hf - ht) / hf


diff_shares = [share(s, "fp32_diff", "ternary_diff") for s in (0, 1, 2)]
ar_shares = [share(s, "fp32_ar", "ternary_ar") for s in (0, 1, 2)]
claimed_diff = [0.2607, 0.2558, 0.2541]
claimed_ar = [0.1523, 0.1405, 0.1436]
for s in range(3):
    check(f"diffusion share seed {s}", diff_shares[s], claimed_diff[s], 0.0005,
          "preprint.md:591")
    check(f"AR share seed {s}", ar_shares[s], claimed_ar[s], 0.0005,
          "preprint.md:592")
dm, am = statistics.fmean(diff_shares), statistics.fmean(ar_shares)
dsd, asd = statistics.stdev(diff_shares), statistics.stdev(ar_shares)  # n-1
print(f"  diffusion: mean={dm:.6f} sd(n-1)={dsd:.6f}")
print(f"  AR:        mean={am:.6f} sd(n-1)={asd:.6f}")
check("diffusion mean share", dm, 0.2569, 0.0005, "preprint.md:591 (abstract 25.7%: :38)")
check("diffusion sd", dsd, 0.0034, 0.0005, "preprint.md:591")
check("AR mean share", am, 0.1455, 0.0005, "preprint.md:592 (abstract 14.5%: :38)")
check("AR sd", asd, 0.0061, 0.0005, "preprint.md:592")
print(f"  abstract percentages: {dm*100:.1f}% vs {am*100:.1f}% "
      f"(claimed 25.7% / 14.5%, preprint.md:38)")

print("\n" + "=" * 78)
print("3. Per-seed interaction, paired SE, ratios")
print("=" * 78)
inter = [d - a for d, a in zip(diff_shares, ar_shares)]
claimed_inter = [0.1084, 0.1153, 0.1105]
for s in range(3):
    check(f"interaction seed {s}", inter[s], claimed_inter[s], 0.0005,
          "preprint.md:595")
im = statistics.fmean(inter)
ise = statistics.stdev(inter) / math.sqrt(3)  # sd uses n-1
print(f"  paired diffs: {['%.6f' % x for x in inter]}")
print(f"  mean={im:.6f}  sd(n-1)={statistics.stdev(inter):.6f}  SE={ise:.6f}")
check("interaction mean", im, 0.1114, 0.0005, "preprint.md:595")
check("paired SE", ise, 0.0020, 0.0005, "preprint.md:595")
print(f"  ratio to SE = {im/ise:.2f}x (claimed 54x, preprint.md:595)")
print(f"  ratio of shares = {dm/am:.4f}x (claimed 1.77x, preprint.md:598)")
check("ratio-to-SE/54", im / ise / 54, 1.0, 0.02, "preprint.md:595")
check("share ratio", dm / am, 1.77, 0.01, "preprint.md:598")

print("\n" + "=" * 78)
print("4. Raw-nats deltas (cell means)")
print("=" * 78)
dn = cell_means["ternary_diff"] - cell_means["fp32_diff"]
an = cell_means["ternary_ar"] - cell_means["fp32_ar"]
check("diffusion delta NELBO", dn, 2.071, 0.001, "preprint.md:601")
check("AR delta NLL", an, 1.344, 0.001, "preprint.md:602")

print("\n" + "=" * 78)
print("5. Budget ladder (seed 0)")
print("=" * 78)
for budget, label, cd, ca, cint in [(1200, "rung2 (4.92M)", 0.1630, 0.0584, 0.1046),
                                    (4800, "rung3 (19.7M)", 0.0884, -0.0697, None)]:
    ds = share(0, "fp32_diff", "ternary_diff", budget)
    as_ = share(0, "fp32_ar", "ternary_ar", budget)
    print(f"  {label}: diff share={ds:.6f} AR share={as_:.6f} interaction={ds - as_:.6f}")
    check(f"{label} diffusion share", ds, cd, 0.0005, "preprint.md:634,657-658")
    check(f"{label} AR share", as_, ca, 0.0005, "preprint.md:634,646")
    if cint is not None:
        check(f"{label} interaction", ds - as_, cint, 0.0005,
              "preprint.md:635 (+0.1046; abstract :50 says +0.105)")
    for _, r in by_budget[budget]:
        print(f"      {r['cell']}: eval={r['loss_nats']:.4f} "
              f"train={r['train']['final_train_loss']:.4f}")

r3 = {r["cell"]: r for _, r in by_budget[4800]}
check("rung3 fp32_ar train loss", r3["fp32_ar"]["train"]["final_train_loss"],
      0.0302, 0.0005, "preprint.md:645")
check("rung3 fp32_ar eval NLL", r3["fp32_ar"]["loss_nats"], 4.8898, 0.001,
      "preprint.md:645")
check("rung3 ternary_ar train loss", r3["ternary_ar"]["train"]["final_train_loss"],
      0.7774, 0.0005, "preprint.md:646")
check("rung3 ternary AR cost (nats)", r3["ternary_ar"]["loss_nats"] - r3["fp32_ar"]["loss_nats"],
      -0.49, 0.005, "preprint.md:647")
r3int = share(0, "fp32_diff", "ternary_diff", 4800) - share(0, "fp32_ar", "ternary_ar", 4800)
check("rung3 nominal interaction", r3int, 0.158, 0.001, "preprint.md:648")
r2 = {r["cell"]: r for _, r in by_budget[1200]}
print(f"  rung2 interaction vs base-seed0 ({inter[0]:.4f}): delta="
      f"{abs((share(0,'fp32_diff','ternary_diff',1200)-share(0,'fp32_ar','ternary_ar',1200))-inter[0]):.4f} "
      f"(rule: <0.03)")
print(f"  ratio widening: base seed0 {diff_shares[0]/ar_shares[0]:.2f}x -> rung2 "
      f"{share(0,'fp32_diff','ternary_diff',1200)/share(0,'fp32_ar','ternary_ar',1200):.2f}x "
      f"(claimed 1.7x -> 2.8x, preprint.md:637-638)")
print(f"  rung3 diffusion train/eval: fp32 {r3['fp32_diff']['train']['final_train_loss']:.2f}/"
      f"{r3['fp32_diff']['loss_nats']:.2f}  ternary {r3['ternary_diff']['train']['final_train_loss']:.2f}/"
      f"{r3['ternary_diff']['loss_nats']:.2f} (claimed 3.61/3.35, 4.38/4.11, preprint.md:654-655)")
mono = [diff_shares[0], share(0, "fp32_diff", "ternary_diff", 1200),
        share(0, "fp32_diff", "ternary_diff", 4800)]
print(f"  diffusion share vs budget: {['%.4f' % x for x in mono]} "
      f"monotone falling: {mono[0] > mono[1] > mono[2]} (claimed 0.2607->0.1630->0.0884)")

print("\n" + "=" * 78)
print("6. Gating numbers (diffusion_ledger.jsonl)")
print("=" * 78)
dl = rows("diffusion_ledger.jsonl")
assert len(dl) == 1 and dl[0]["arm"] == "diffusion_fp32"
g = dl[0]
nb, na = g["nelbo_before"], g["nelbo_after"]
check("NELBO before", nb["nelbo"], 10.539, 0.0005, "preprint.md:32,544")
check("NELBO after", na["nelbo"], 3.876, 0.0005, "preprint.md:32,545 (grid row: 3.8761)")
check("mask acc before", nb["mask_accuracy"], 0.008, 0.0005, "preprint.md:33,544")
check("mask acc after", na["mask_accuracy"], 0.275, 0.0005, "preprint.md:33,545")
check("unadapted headroom below floor", FLOOR - nb["nelbo"], 1.39, 0.005,
      "preprint.md:547")
check("gain (nats)", nb["nelbo"] - na["nelbo"], 6.66, 0.005, "preprint.md:549 / ledger gain_nats")
check("after headroom", FLOOR - na["nelbo"], 8.06, 0.005, "preprint.md:32,549")
gridrow = base[("fp32_diff", 0)][1]
print(f"  reproduction across scripts: diffusion_ledger after={na['nelbo']:.10f} vs "
      f"grid fp32_diff seed0={gridrow['loss_nats']:.10f} "
      f"identical={na['nelbo'] == gridrow['loss_nats']}")

print("\n" + "=" * 78)
print("7. Cross-step compounding (compounding_ledger.jsonl)")
print("=" * 78)
comp = rows("compounding_ledger.jsonl")
assert len(comp) == 8
tab = {(r["arm"], r["epsilon"]): r for r in comp}
print("  arm/eps: per-forward argmax, S=1, S=32, growth")
growths = []
for arm in ("ternary", "fp32_control"):
    for eps in (1e-6, 1e-4, 1e-3, 1e-2):
        r = tab[(arm, eps)]
        pf = r["per_forward"]["argmax_disagreement"]
        s1 = r["disagreement_by_steps"]["1"]
        s32 = r["disagreement_by_steps"]["32"]
        gr = s32 / s1 if s1 > 0 else float("nan")
        if s1 > 0 and s32 > 0:
            growths.append(((arm, eps), gr))
        print(f"    {arm:13s} eps={eps:g}: pf={pf:.6f} S1={s1:.6f} S32={s32:.6f} "
              f"growth={gr:.2f}x")
print(f"  amplification range over nonzero cells: "
      f"{min(g for _, g in growths):.2f}x .. {max(g for _, g in growths):.2f}x "
      f"(claimed ~6-8x, preprint.md:729-732; table :745-752 claims 7.9/6.0/6.1/8.2/7.9)")
claimed_growth = {("ternary", 1e-4): 7.9, ("ternary", 1e-3): 6.0, ("ternary", 1e-2): 6.1,
                  ("fp32_control", 1e-3): 8.2, ("fp32_control", 1e-2): 7.9}
for k2, cg in claimed_growth.items():
    gr = tab[k2]["disagreement_by_steps"]["32"] / tab[k2]["disagreement_by_steps"]["1"]
    check(f"growth {k2[0]} eps={k2[1]:g}", gr, cg, 0.05, "preprint.md:745-752")

t4 = tab[("ternary", 1e-4)]
f4 = tab[("fp32_control", 1e-4)]
check("ternary eps=1e-4 flipped levels", t4["flips"]["flipped_levels"], 7260, 0.5,
      "preprint.md:747")
check("total levels (~308M)", t4["flips"]["total_levels"] / 1e6, 308.281344, 0.001,
      "preprint.md:747 '308M' (=308,281,344)")
check("flip fraction 2.4e-5", t4["flips"]["flip_fraction"] * 1e5, 2.354991679, 0.05,
      "preprint.md:747")
check("ternary per-forward 5.3%", t4["per_forward"]["argmax_disagreement"] * 100, 5.3,
      0.05, "preprint.md:747-748")
check("fp32 control per-forward 0.26%", f4["per_forward"]["argmax_disagreement"] * 100,
      0.26, 0.005, "preprint.md:748,757")
ratio = t4["per_forward"]["argmax_disagreement"] / f4["per_forward"]["argmax_disagreement"]
print(f"  ternary/fp32 per-forward ratio at eps=1e-4: {ratio:.1f}x (claimed 20x, preprint.md:748)")
check("healed: fp32 eps=1e-4 S=32 == 0", f4["disagreement_by_steps"]["32"], 0.0, 1e-12,
      "preprint.md:756-757 'ends at exactly zero'")
print(f"    NOTE fp32 eps=1e-4 full trajectory: {f4['disagreement_by_steps']} "
      f"(S=16 is {f4['disagreement_by_steps']['16']:.4f}, nonzero)")
t6 = tab[("ternary", 1e-6)]
print(f"  ternary eps=1e-6: {t6['flips']['flipped_levels']} flipped levels, all steps "
      f"{list(t6['disagreement_by_steps'].values())} (claimed 83 flips absorbed, preprint.md:761)")
print(f"  42% trajectory divergence: ternary eps=1e-4 S32={t4['disagreement_by_steps']['32']*100:.1f}% "
      f"(claimed 42%, preprint.md:751); fp32 needs eps=1e-2 "
      f"(S32={tab[('fp32_control',1e-2)]['disagreement_by_steps']['32']*100:.1f}%) -> "
      f"eps ratio 1e-2/1e-4 = 100x = two orders of magnitude (preprint.md:752-753)")

print("\n" + "=" * 78)
print("8. Inference bench (inference_bench.jsonl)")
print("=" * 78)
bench = rows("inference_bench.jsonl")
ar = {r["variant"]: r for r in bench if r["mode"] == "ar_greedy_kv_cache"}
diff = {(r["variant"], r["denoise_steps"]): r for r in bench if r["mode"] == "diffusion_denoise"}
ar_rate = ar["fp32"]["tokens_per_s"]
check("FP32 AR KV-cached tok/s", ar_rate, 42.6, 0.05, "SUMMARY.md:104 / preprint.md:772")
qrates = [ar[v]["tokens_per_s"] for v in ("q1_0_dequant", "q1_0_int8", "q1_58_dequant", "q1_58_int8")]
print(f"  quantized AR tok/s: {qrates} -> range {min(qrates):.2f}-{max(qrates):.2f} "
      f"(claimed 4.8-6.0, SUMMARY.md:105 / preprint.md:773)")
for steps, claimed_rate, claimed_ratio in [(1, 1885.8, 44.3), (8, 238.7, 5.6),
                                           (16, 122.6, 2.9), (32, 60.0, 1.4)]:
    r = diff[("fp32_diff", steps)]
    check(f"fp32_diff @{steps} tok/s", r["tokens_per_s"], claimed_rate, 0.05,
          "preprint.md:790-795 / SUMMARY.md:106-107")
    check(f"fp32_diff @{steps} vs AR", r["tokens_per_s"] / ar_rate, claimed_ratio, 0.05,
          "preprint.md:790-795")
check("ternary diff @32 tok/s", diff[("q1_58_diff", 32)]["tokens_per_s"], 16.1, 0.05,
      "SUMMARY.md:108 / preprint.md:808")
per_step = [diff[("fp32_diff", s)]["median_s"] / s for s in (1, 4, 8, 16, 32)]
print(f"  per-denoise-step cost (median_s/steps): {['%.4f' % c for c in per_step]} "
      f"(claimed flat ~0.067 s, preprint.md:786 / SUMMARY.md:113)")
mean_step = statistics.fmean(per_step)
crossover = 128 / (ar_rate * mean_step)
crossover_1step = 128 / (ar_rate * per_step[0])
crossover_32 = 32 * diff[("fp32_diff", 32)]["tokens_per_s"] / ar_rate
print(f"  crossover S where 128/(c*S)=AR rate: mean-c -> {crossover:.1f}, "
      f"1-step-c -> {crossover_1step:.1f}, scaling from 32-step rate -> {crossover_32:.1f} "
      f"(claimed ~45, preprint.md:67,794 / SUMMARY.md:115)")
check("crossover ~45", crossover, 45, 2.0, "preprint.md:794")
print(f"  ternary/fp32 diffusion penalty @32: {diff[('fp32_diff',32)]['tokens_per_s']/diff[('q1_58_diff',32)]['tokens_per_s']:.2f}x "
      f"(claimed 3-4x, preprint.md:808-810)")

print("\nDone. Any line flagged MISMATCH above is a genuine discrepancy.")
