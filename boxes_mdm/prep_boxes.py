#!/usr/bin/env python3
"""Convert Kim & Schuster boxes data -> diffusion-vs-ar format (word-level vocab).

usage: python prep_boxes.py --boxes_dir /tmp/boxes/boxes-dataset-v1 --repo ~/diffusion-vs-ar \
           [--split t5_boxes_nso_exp2_max3] [--prefix boxes]
"""
import argparse, glob, importlib.util, json, os, re, shutil

ap = argparse.ArgumentParser()
ap.add_argument("--boxes_dir", required=True)
ap.add_argument("--repo", required=True)
ap.add_argument("--split", default="t5_boxes_nso_exp2_max3")
ap.add_argument("--prefix", default="boxes")  # must not contain 'cd' or 'path' (metric.py substring checks)
a = ap.parse_args()
assert "cd" not in a.prefix and "path" not in a.prefix

def words(s):  # "a, b." -> ["a", ",", "b", "."]
    return re.sub(r"([,.])", r" \1 ", s).split()

def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

# 1. vocab = every word in every t5_* split (so one tokenizer works for all splits)
vocab = set()
for f in glob.glob(os.path.join(a.boxes_dir, "t5_*", "*.jsonl")):
    for r in load(f):
        vocab.update(words(r["sentence"]))
vocab = sorted(vocab) + [" "]  # ' ' must be a token: CustomTokenizer emits a ' ' between words
print("word vocab:", len(vocab) - 1, "+ space")

# 2. model/tokenizer configs with the new vocab (specials take ids 0-4, vocab starts at 5)
for src in ["model_config_tiny", "model_config", "model_config_medium"]:
    dst = os.path.join(a.repo, src + "_" + a.prefix)
    shutil.copytree(os.path.join(a.repo, src), dst, dirs_exist_ok=True)
    tc = json.load(open(os.path.join(dst, "tokenizer_config.json")))
    tc["vocab"] = vocab
    tc["model_max_length"] = 1024
    json.dump(tc, open(os.path.join(dst, "tokenizer_config.json"), "w"), indent=4)
    cc = json.load(open(os.path.join(dst, "config.json")))
    cc["vocab_size"] = len(vocab) + 5
    cc["n_positions"] = cc["n_ctx"] = max(cc.get("n_positions", 512), 512)
    json.dump(cc, open(os.path.join(dst, "config.json"), "w"), indent=2)

# 3. tokenizer for length stats (load custom_tokenizer.py directly, no torch needed)
spec = importlib.util.spec_from_file_location(
    "ct", os.path.join(a.repo, "src/llmtuner/tuner/core/custom_tokenizer.py"))
ct = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct)
tok = ct.CustomTokenizer(vocab, model_max_length=1024)

# 4. write data
os.makedirs(os.path.join(a.repo, "data"), exist_ok=True)
info_path = os.path.join(a.repo, "data", "dataset_info.json")
info = json.load(open(info_path)) if os.path.exists(info_path) else {}
max_len = max_tgt = 0
for sp, fn in [("train", "train-t5.jsonl"), ("dev", "dev-t5.jsonl"), ("test", "test-t5.jsonl")]:
    rows = load(os.path.join(a.boxes_dir, a.split, fn))
    out, meta = [], []
    for r in rows:
        pre, _ = r["sentence_masked"].split(" <extra_id_0>")
        prompt = " ".join(words(pre))                       # ends with "Box k"
        resp = r["masked_content"].replace("<extra_id_0>", "").strip()   # "contains the a and the b" | "is empty"
        out.append({"prompt": prompt, "response": resp})
        meta.append({"numops": r["numops"], "gold": resp,
                     "context": r["sentence_masked"].replace("<extra_id_0> .", "."),
                     "sentence": r["sentence"]})
        s, t = tok.encode(prompt), tok.encode(resp)
        assert 4 not in s + t, "UNK token produced"
        max_len = max(max_len, len(s) + 1 + len(t) + 1); max_tgt = max(max_tgt, len(t) + 1)
    name = f"{a.prefix}_{sp}"
    with open(os.path.join(a.repo, "data", name + ".jsonl"), "w") as f:
        f.write("\n".join(json.dumps(o) for o in out))
    with open(os.path.join(a.repo, "data", name + ".meta.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(m) for m in meta))
    info[name] = {"file_name": name + ".jsonl", "columns": {"prompt": "prompt", "response": "response"}}
    print(name, len(out))
json.dump(info, open(info_path, "w"), indent=2)
print(f"\nCUTOFF_LEN >= {max_len}   (use {((max_len + 7)//8)*8})")
print(f"AR max_new_tokens >= {max_tgt}   (use {max_tgt + 6})")
json.dump({"cutoff_len": ((max_len + 7)//8)*8, "max_new_tokens": max_tgt + 6},
          open(os.path.join(a.repo, "data", a.prefix + "_lengths.json"), "w"))
