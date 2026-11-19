# usage: python patch_device.py <path to diffusion-vs-ar>
# Makes the upstream code run without a GPU (CPU) and on any single CUDA device: replaces hardcoded
# .cuda() calls with the active device. Idempotent: safe to run more than once.
import sys
root = sys.argv[1]
EDITS = {
    "src/llmtuner/tuner/mdm/trainer.py": [
        ("        self.model.cuda()\n", "        self.model.to(self.args.device)\n"),
        ("        x = inputs['input_ids'].cuda()\n", "        x = inputs['input_ids'].to(self.args.device)\n"),
        ("        src_mask = inputs['src_mask'].bool().cuda()\n", "        src_mask = inputs['src_mask'].bool().to(self.args.device)\n"),
    ],
    "src/llmtuner/tuner/core/loader.py": [
        ("map_location=torch.device('cuda')", "map_location=torch.device('cpu')"),
    ],
    "src/llmtuner/tuner/mdm/workflow.py": [
        ("        if not dist.is_initialized():\n            dist.init_process_group(backend='nccl')\n",
         "        if int(__import__('os').environ.get('WORLD_SIZE', '1')) > 1 and not dist.is_initialized():  # only when launched distributed\n"
         "            dist.init_process_group(backend='nccl')\n"),
    ],
    "src/llmtuner/extras/misc.py": [
        ("        return model.cuda()\n", "        return model.to('cuda' if torch.cuda.is_available() else 'cpu')\n"),
    ],
}
for rel, pairs in EDITS.items():
    p = f"{root}/{rel}"
    s = open(p).read()
    changed = False
    for old, new in pairs:
        if new in s:
            continue
        assert s.count(old) == 1, f"{rel}: expected exactly one occurrence of {old.strip()!r} (upstream changed?)"
        s = s.replace(old, new); changed = True
    open(p, "w").write(s)
    print(("patched " if changed else "already patched ") + p)
