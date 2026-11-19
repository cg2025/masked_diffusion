# masked_diffusion: boxes entity tracking, MDM vs AR

Glue code that runs the Kim & Schuster "boxes" task through
[diffusion-vs-ar](https://github.com/HKUNLP/diffusion-vs-ar) (masked diffusion LM vs autoregressive baseline).
Both upstream repos are git submodules (pinned commits).

## One-time setup (cluster)
```bash
git clone --recursive https://github.com/cg2025/masked_diffusion.git /data/scratch/cgoyal/boxes-diffusion
cd /data/scratch/cgoyal/boxes-diffusion
# if already cloned without --recursive:  git submodule update --init
mkdir -p logs                                   # sbatch --output dir must exist
bash boxes_mdm/make_env.sh                      # login node: builds .venv (py3.9, torch 1.13.1+cu117)
source .venv/bin/activate && bash boxes_mdm/setup_boxes.sh   # builds data/boxes_*, patches metric.py, copies scripts
```
The Google Drive data from the diffusion-vs-ar README is **not** needed for boxes.

## Run
```bash
sbatch boxes-smoke.sbatch                                            # 2-epoch MDM smoke test, 1 GPU
cd diffusion-vs-ar && NGPU=8 EPOCHS=100 bash scripts/boxes/train-mdm.sh   # full MDM
cd diffusion-vs-ar && NGPU=8 EPOCHS=100 bash scripts/boxes/train-sft.sh   # AR baseline
```
Results: `python scripts/boxes/score_boxes.py --pred <exp>/boxes_test/generated_predictions.jsonl --meta data/boxes_test.meta.jsonl`

## Notes
- `setup_boxes.sh` and `patch_metric.py` are idempotent; they modify the `diffusion-vs-ar` working tree (shows as dirty submodule). Don't commit the generated data (dataset authors' request).
- The smoke job excludes node `gpu20-2`, where earlier jobs saw no GPU. Check with `srun --partition=drl-titanrtx --qos=drl-main --account=drl --gres=gpu:1 nvidia-smi -L`.
