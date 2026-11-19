#!/usr/bin/env python3
"""Score generated_predictions.jsonl on boxes_test; accuracy by #ops on the queried box.
Also writes a TSV in the format of entity-tracking-lms/src/evaluation/compute_metrics.py (--tsv)."""
import argparse, collections, json

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True)
ap.add_argument("--meta", required=True)
ap.add_argument("--tsv", default=None)
a = ap.parse_args()

def items(x):
    x = x.strip()
    if x == "is empty": return frozenset()
    if not x.startswith("contains "): return None
    return frozenset(i.replace("the ", "", 1).strip() for i in x[len("contains "):].split(" and "))

preds = [json.loads(l) for l in open(a.pred) if l.strip()]
meta = [json.loads(l) for l in open(a.meta) if l.strip()]
assert len(preds) == len(meta), f"{len(preds)} preds vs {len(meta)} meta (use --max_samples consistently)"
tot, cor = collections.Counter(), collections.Counter()
rows = []
for p, m in zip(preds, meta):
    pi, gi = items(p["predict"]), items(m["gold"])
    ok = pi is not None and pi == gi
    for k in (m["numops"], "all"):
        tot[k] += 1; cor[k] += ok
    rows.append((m["gold"], p["predict"].strip(), m["context"]))
print("numops  n      acc")
for k in sorted(k for k in tot if k != "all"):
    print(f"{k:<7} {tot[k]:<6} {cor[k]/tot[k]:.4f}")
print(f"all     {tot['all']:<6} {cor['all']/tot['all']:.4f}")
if a.tsv:
    with open(a.tsv, "w") as f:
        f.write("\ttarget\tprediction\tinput\n")
        for i, (g, p, c) in enumerate(rows): f.write(f"{i}\t{g}\t{p}\t{c}\n")
