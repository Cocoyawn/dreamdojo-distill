# DreamDojo Distillation Package

This package contains the DreamDojo piper distillation code/config changes and SLURM scripts used for teacher generation, warmup distillation, and PSNR evaluation.

External assets are intentionally not committed:
- teacher checkpoint: https://huggingface.co/Shirk6/dreamdojo-piper-vertical-1440-640-fps10-iter30000
- expert dataset: https://huggingface.co/datasets/m1ku2/battery_assemble

Entry points:
- `codex_jobs/codex_setup_dreamdojo.sbatch`
- `codex_jobs/codex_teacher_gen_piper.sbatch`
- `codex_jobs/codex_warmup_piper.sbatch`
- `codex_jobs/codex_eval_teacher_piper.sbatch`
- `scripts/codex_psnr_curve.py`

Do not commit credentials. Put W&B/GitHub/HF tokens in local env files outside this repo.
