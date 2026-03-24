# OpenVLA 数据采集启动说明

本文档用于快速启动和管理量化数据采集任务。

## 1) 采集目标（当前默认）
- 使用 GPU 2 和 GPU 3 并行采集
- 每条配置只跑前 5 个任务
- 每个任务只跑 1 次（`num_trials_per_task=1`）
- 不保存回放视频，只保存指标数据和日志

## 2) 先进入目录
```bash
cd /mnt/hdd0/zilai_wan22_test/models/openvla_libero_quant/openvla
```

## 3) 启动采集（tmux 后台）
```bash
tmux kill-session -t libero_sweep >/dev/null 2>&1 || true

tmux new-session -d -s libero_sweep "bash -lc '
cd /mnt/hdd0/zilai_wan22_test/models/openvla_libero_quant/openvla && \
PYTHONPATH=. stdbuf -oL -eL /home/zilai/anaconda3/envs/libero/bin/python \
experiments/robot/libero/run_libero_quant_sweep.py \
  --repo_root . \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 1 \
  --max_tasks_per_run 5 \
  --save_rollout_videos False \
  --quant_modes int4,int8 \
  --llm_targets ffn_only,all,attn_only,none \
  --llm_ratios 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --llm_layer_selections prefix,suffix,uniform \
  --vision_targets none,projector_only,tower_only,all \
  --action_targets none,all \
  --seeds 7,17,27 \
  --repeat_per_config 3 \
  --max_unique_configs 500 \
  --gpu_ids 2,3 \
  --resume False \
  --output_dir /mnt/hdd0/zilai_wan22_test/models/openvla_libero_quant/outputs \
  2>&1 | tee /mnt/hdd0/zilai_wan22_test/models/openvla_libero_quant/outputs/sweep_progress.log
'"
```

## 4) 查看进度
### 方式 A：看总进度日志（推荐）
```bash
tail -f /mnt/hdd0/zilai_wan22_test/models/openvla_libero_quant/outputs/sweep_progress.log
```

### 方式 B：看结构化进度 JSON
```bash
watch -n 2 'cat /mnt/hdd0/zilai_wan22_test/models/openvla_libero_quant/outputs/progress.json'
```

### 方式 C：看已完成条数
```bash
watch -n 5 'ls /mnt/hdd0/zilai_wan22_test/models/openvla_libero_quant/outputs/*.json 2>/dev/null | wc -l'
```

## 5) 停止采集
```bash
tmux kill-session -t libero_sweep
```

## 6) 输出文件位置
- 每条采集数据（JSON）:
  - `/mnt/hdd0/zilai_wan22_test/models/openvla_libero_quant/outputs/*.json`
- 总进度日志:
  - `/mnt/hdd0/zilai_wan22_test/models/openvla_libero_quant/outputs/sweep_progress.log`
- 结构化进度:
  - `/mnt/hdd0/zilai_wan22_test/models/openvla_libero_quant/outputs/progress.json`
- 单条运行日志:
  - `/mnt/hdd0/zilai_wan22_test/models/openvla_libero_quant/outputs/run_logs/`

## 7) 快速自检
```bash
tmux ls | grep libero_sweep
ps -ef | grep 'run_libero_eval.py' | grep -v grep
nvidia-smi
```
