# masked_diffusion: boxes entity tracking, MDM vs AR

Glue code that runs the Kim & Schuster "boxes" task through
[diffusion-vs-ar](https://github.com/HKUNLP/diffusion-vs-ar) (masked diffusion LM vs autoregressive baseline).
Both upstream repos are git submodules (pinned commits). A GPU is **optional**: the model is tiny, so CPU works
(slowly); a single GPU (e.g. a free Colab T4) is much faster.

## Run it anywhere
```bash
git clone --recursive https://github.com/cg2025/masked_diffusion.git && cd masked_diffusion
pip install -r requirements.txt
QUICK=1 bash boxes_mdm/run_boxes.sh          # sanity run: 4000 train / 500 test examples, 3 epochs
bash boxes_mdm/run_boxes.sh                  # full run (100 epochs on GPU; set EPOCHS=... on CPU)
MODE=ar bash boxes_mdm/run_boxes.sh          # autoregressive baseline
```
Colab: open `boxes_colab.ipynb`. `run_boxes.sh` auto-detects CPU / 1 GPU / several GPUs, prepares the data on first run
(no Google Drive download needed), trains, evaluates and prints accuracy by number of operations.
Knobs: `MODE EPOCHS MODEL LR BS T MAX_TRAIN MAX_TEST VAL_SIZE NGPU QUICK`.

## What the scripts do
- `boxes_mdm/setup_boxes.sh`: unzips the boxes data, builds the word-level vocab/configs (`prep_boxes.py`), and applies two idempotent patches to the upstream checkout: `patch_metric.py` (boxes accuracy metric) and `patch_device.py` (removes hardcoded `.cuda()` / NCCL init so CPU and single-GPU work).
- `boxes_mdm/run_boxes.sh`: train, evaluate and score (`scripts_boxes/score_boxes.py`).
- `boxes-smoke.sbatch` and `boxes_mdm/make_env.sh`: optional Slurm wrapper / uv environment builder for a cluster.

## Notes
- The patches modify the `diffusion-vs-ar` working tree (shows as a dirty submodule). Don't commit the generated data (dataset authors' request).
- On a cluster, `mkdir -p logs` before `sbatch`, and check GPU allocation with `srun --gres=gpu:1 nvidia-smi -L`. Earlier jobs saw no GPU on node `gpu20-2`, which the sbatch excludes.
