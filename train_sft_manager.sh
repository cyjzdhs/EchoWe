#!/bin/bash
set -e

TIME=$(date +"%m%d_%H%M")
LOG_FILE="output/logs/sft_${TIME}.log"
REPORT_DIR="output/reports/sft_${TIME}"
PID_FILE="output/train_sft.pid"

mkdir -p output/logs
mkdir -p output/reports

echo "=================================="
echo "SFT Start: $(date)"
echo "Log: $LOG_FILE"
echo "Trainer output_dir: ./wechat_lora (fixed)"
echo "Report: $REPORT_DIR"
echo "=================================="

nohup python -u train_sft.py > "$LOG_FILE" 2>&1 &
PID=$!
echo $PID > "$PID_FILE"

echo "PID: $PID"
echo "tail -f $LOG_FILE"
echo "watch -n 1 nvidia-smi"
echo "=================================="
echo "Running in background..."

# 轮询等待结束，然后出报告（不用 wait，避免 not-a-child）
(
  while kill -0 $PID 2>/dev/null; do
    sleep 10
  done

  echo "SFT Training finished. Generating report..."
  mkdir -p "$REPORT_DIR"

  python - <<EOF
import os, glob, json, csv

OUT="wechat_lora"
REP="$REPORT_DIR"

ck=glob.glob(os.path.join(OUT,"checkpoint-*"))
if not ck:
    raise SystemExit("No checkpoints found in wechat_lora/")
ck=sorted(ck, key=lambda x:int(x.split("-")[-1]))
latest=ck[-1]
state=os.path.join(latest,"trainer_state.json")

with open(state,"r") as f:
    st=json.load(f)

steps=[]
loss=[]
for it in st.get("log_history",[]):
    if "loss" in it and "step" in it:
        steps.append(it["step"])
        loss.append(it["loss"])

csvp=os.path.join(REP,"loss_history.csv")
with open(csvp,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["step","loss"])
    for s,l in zip(steps,loss):
        w.writerow([s,l])

last20=loss[-20:] if len(loss)>=20 else loss
summary=f"""===== SFT Summary =====
trainer_state: {state}
points: {len(loss)}
last_step: {steps[-1] if steps else 0}
final_loss: {loss[-1] if loss else 0}
min_loss: {min(loss) if loss else 0}
avg_last20: {sum(last20)/len(last20) if last20 else 0}
"""
with open(os.path.join(REP,"summary.txt"),"w") as f:
    f.write(summary)

try:
    import matplotlib.pyplot as plt
    plt.plot(steps, loss)
    plt.title("SFT Loss Curve")
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.savefig(os.path.join(REP,"loss_curve.png"))
    plt.close()
except:
    pass

print(summary)
EOF

  echo "Report saved to: $REPORT_DIR"
) &