export PYTHONPATH=/mnt/amlfs-01/shared/shenyuang/DreamDojo:$PYTHONPATH
export HF_HOME=/mnt/amlfs-01/shared/shenyuang/cosmos_cache
source /mnt/amlfs-01/shared/shenyuang/DreamDojo/.venv/bin/activate

python examples/action_conditioned.py \
  -o outputs/action_conditioned/basic \
  --checkpoints-dir /mnt/amlfs-01/shared/shenyuang/cosmos_logs/exp1201/gr1/checkpoints \
  --experiment dreamdojo_2b_480_640_gr1 \
  --save-dir /mnt/amlfs-01/shared/shenyuang/dreamdojo_results/gr1_unified_test \
  --num-frames 49 \
  --num-samples 100 \
  --dataset-path datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot \
  --data-split test \
  --deterministic-uniform-sampling \
  --checkpoint-interval 5000 \
  --infinite