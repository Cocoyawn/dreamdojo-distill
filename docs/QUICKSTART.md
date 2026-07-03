# 远端启动 warmup 蒸馏训练

三步启动，其他都是等待。

## 1. Clone + 环境

```bash
git clone https://github.com/Cocoyawn/dreamdojo-distill.git
cd dreamdojo-distill

HF_TOKEN=hf_xxxxxxxxxxxx bash local_jobs/setup_remote.sh
```

`setup_remote.sh` 会：
- 装依赖（`.venv/`）
- 下载数据集 210 GB → `datasets/piper_warmup_regenerated_4step/`
- 下载 teacher ckpt 17 GB → `checkpoints/dreamdojo-piper-vertical-1440-640-fps10-iter30000/`（warmup 不用，SF 阶段用）
- 下载 cr1 empty-string embedding
- 跑 `verify_dataset.py` 校验

**HF token 必须先在 https://huggingface.co/nvidia/Cosmos-Predict2.5-2B 接受 license**，否则拉不到 cr1 embedding。

**要求**：≥ 8×80 GB GPU，≥ 450 GB 磁盘。

## 2. Dry-run 摸边界（15 分钟）

```bash
bash local_jobs/warmup_launch.sh dryrun
```

跑 200 iter，wandb offline。**看两个数字**：
- `sec/iter`：15-20 s 优秀，30 s 中等，> 60 s 有问题
- `loss`：应从 ~1.0 降到 ~0.3；卡住不动 → cr1 embedding 有问题

OOM 就说明 8 GPU × BATCH=1 都装不下，要检查 config，不是加卡能解决。

## 3. 正式训练

```bash
tmux new -d -s dd-warmup 'bash local_jobs/warmup_launch.sh full 2>&1 | tee logs/warmup_full.log'
tmux attach -t dd-warmup   # Ctrl-b d 挂回后台
```

默认：**8 GPU × batch=2 × 10000 iter，每 1000 iter 存一次**（10 个 ckpt，共 ~150 GB）。

预计 wall-clock：
- 乐观 ~2 天（sec/iter 18-20）
- 中等 ~3-4 天（sec/iter 25-35）
- 悲观 ~6-7 天（sec/iter 50-60）

Ckpt 落在 `dreamdojo_logs/cosmos_interactive/interactive_warmup/<RUN_NAME>/checkpoints/iter_00000<n>000/`。

## 参数覆盖

三档模式默认参数在 `local_jobs/warmup_launch.sh` 里；用环境变量覆盖：

```bash
NPROC=4 bash local_jobs/warmup_launch.sh full           # 4 卡（默认 8）
BATCH=4 bash local_jobs/warmup_launch.sh full           # 加大 batch（如果显存有余）
MAX_ITER=5000 bash local_jobs/warmup_launch.sh full     # 早停
SAVE_ITER=2000 bash local_jobs/warmup_launch.sh full    # 稀释 ckpt（省磁盘）
WANDB_MODE=offline bash local_jobs/warmup_launch.sh full  # 无 wandb
```

或直接传 hydra override：
```bash
bash local_jobs/warmup_launch.sh full dataloader_train.batch_size=4 trainer.save_iter=500
```

## 上传训好的 ckpt

```bash
CKPT=dreamdojo_logs/cosmos_interactive/interactive_warmup/<RUN_NAME>/checkpoints/iter_000010000
HF_HUB_ENABLE_HF_TRANSFER=1 hf upload \
  Cocoyawn32/dreamdojo-piper-warmup-v2 "$CKPT" . \
  --repo-type=model
```

## 常见问题

| 症状 | 原因 | 处理 |
|---|---|---|
| `datasets/cr1_..._embeddings.pt: No such file` | cr1 没下 | `bash local_jobs/setup_remote.sh SKIP_DEPS=1 SKIP_DATASET=1 SKIP_TEACHER=1` |
| `HTTPError: 401` / `403` | HF token 无效或没接受 nvidia gate | 重设 `HF_TOKEN`；先访问 https://huggingface.co/nvidia/Cosmos-Predict2.5-2B 接受 |
| Loss 一直 ~1.0 不降 | cr1 embedding 加载成 0 | `python -c "import torch; e=torch.load('datasets/cr1_empty_string_text_embeddings.pt'); print(e.norm())"` — 应输出 ~7000 |
| OOM 在 iter 0 | 显存/config 问题 | `BATCH=1 bash local_jobs/warmup_launch.sh dryrun` 再看 |
| `sec/iter` 极慢 + GPU util < 30% | 存储 IO 瓶颈 | 数据集移到本地 NVMe |
| NCCL timeout | 卡间链路慢 | `export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600` |

## 相关

- 代码 https://github.com/Cocoyawn/dreamdojo-distill
- 数据 https://huggingface.co/datasets/Cocoyawn32/dreamdojo-piper-warmup-4step
- Teacher https://huggingface.co/Shirk6/dreamdojo-piper-vertical-1440-640-fps10-iter30000
- 详细文档 `docs/REMOTE_LAUNCH.md`
