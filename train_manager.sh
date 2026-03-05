#!/bin/bash

set -e

TIME=$(date +"%m%d_%H%M")
LOG_FILE="output/logs/train_${TIME}.log"
REPORT_DIR="output/reports/report_${TIME}"
PID_FILE="output/train.pid"

echo "=================================="
echo "Start Training"
echo "Time: $(date)"
echo "Log file: $LOG_FILE"
echo "Trainer output_dir: ./wechat_lora"
echo "=================================="

mkdir -p output/logs
mkdir -p output/reports

# 启动训练（后台）
nohup python -u train_final.py > "$LOG_FILE" 2>&1 &
PID=$!

echo $PID > "$PID_FILE"

echo "Training PID: $PID"
echo "tail -f $LOG_FILE"
echo "watch -n 1 nvidia-smi"
echo "=================================="
echo "Training running in background..."

# 后台监控线程（不使用 wait）
(
while kill -0 $PID 2>/dev/null; do
    sleep 10
done

echo "Training finished."
echo "Generating report..."

mkdir -p "$REPORT_DIR"

python <<EOF
import os, json, glob, csv

OUTPUT_DIR = "wechat_lora"
REPORT_DIR = "$REPORT_DIR"

checkpoints = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
if not checkpoints:
    raise Exception("No checkpoint found.")

checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
latest = checkpoints[-1]

trainer_state_path = os.path.join(latest, "trainer_state.json")

with open(trainer_state_path, "r") as f:
    state = json.load(f)

losses = []
steps = []

for log in state.get("log_history", []):
    if "loss" in log and "step" in log:
        losses.append(log["loss"])
        steps.append(log["step"])

csv_path = os.path.join(REPORT_DIR, "loss_history.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "loss"])
    for s, l in zip(steps, losses):
        writer.writerow([s, l])

summary = f"""
===== Training Summary =====
Total Steps: {steps[-1]}
Final Loss: {losses[-1]}
Minimum Loss: {min(losses)}
Average Last 20 Loss: {sum(losses[-20:]) / min(len(losses),20)}
"""

with open(os.path.join(REPORT_DIR, "summary.txt"), "w") as f:
    f.write(summary)

try:
    import matplotlib.pyplot as plt
    plt.plot(steps, losses)
    plt.title("Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(REPORT_DIR, "loss_curve.png"))
    plt.close()
except:
    pass

print(summary)
EOF

echo "Report saved to: $REPORT_DIR"

) &