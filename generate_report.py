import json
import os
import csv
import matplotlib.pyplot as plt
"""
import glob

checkpoints = glob.glob("wechat_lora/checkpoint-*/trainer_state.json")

if not checkpoints:
    raise FileNotFoundError("未找到 trainer_state.json")
    trainer_path = sorted(checkpoints)[-1]
"""
# ==========================
# 路径 find . -name "trainer_state.json"
# ==========================
trainer_path = "wechat_lora/checkpoint-25000/trainer_state.json"
output_dir = "training_report"

os.makedirs(output_dir, exist_ok=True)

# ==========================
# 读取数据
# ==========================
with open(trainer_path, "r") as f:
    data = json.load(f)

log_history = data["log_history"]

losses = []
lrs = []
steps = []

for i, log in enumerate(log_history):
    if "loss" in log:
        losses.append(log["loss"])
        lrs.append(log.get("learning_rate", 0))
        steps.append(len(steps))

# ==========================
# 统计信息
# ==========================
final_loss = losses[-1]
min_loss = min(losses)
avg_last_20 = sum(losses[-20:]) / len(losses[-20:])
total_steps = len(losses)

# ==========================
# 保存 CSV
# ==========================
csv_path = os.path.join(output_dir, "training_log.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "loss", "learning_rate"])
    for i in range(total_steps):
        writer.writerow([i, losses[i], lrs[i]])

# ==========================
# 画 Loss 曲线
# ==========================
plt.figure(figsize=(10,5))
plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid()
plt.savefig(os.path.join(output_dir, "loss_curve.png"))
plt.close()

# ==========================
# 画 Learning Rate 曲线
# ==========================
plt.figure(figsize=(10,5))
plt.plot(lrs)
plt.title("Learning Rate Curve")
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.grid()
plt.savefig(os.path.join(output_dir, "learning_rate_curve.png"))
plt.close()

# ==========================
# 自动判断状态
# ==========================
if avg_last_20 < 0.75:
    status = "⚠ 可能开始过拟合"
elif 0.8 <= avg_last_20 <= 1.1:
    status = "✅ 人格训练区间理想"
else:
    status = "ℹ 正常训练区间"

# ==========================
# 保存 Summary
# ==========================
summary_path = os.path.join(output_dir, "summary.txt")

with open(summary_path, "w") as f:
    f.write("===== Training Summary =====\n")
    f.write(f"Total Steps: {total_steps}\n")
    f.write(f"Final Loss: {final_loss:.4f}\n")
    f.write(f"Minimum Loss: {min_loss:.4f}\n")
    f.write(f"Average Last 20 Loss: {avg_last_20:.4f}\n")
    f.write(f"Status: {status}\n")

# ==========================
# 输出到终端
# ==========================
print("===== Training Summary =====")
print("Total Steps:", total_steps)
print("Final Loss:", round(final_loss, 4))
print("Minimum Loss:", round(min_loss, 4))
print("Average Last 20 Loss:", round(avg_last_20, 4))
print("Status:", status)

print("\nReport generated in folder:", output_dir)