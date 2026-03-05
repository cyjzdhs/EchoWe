import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "/workspace/user-data/swift_finetune/models/DeepSeek-R1-0528-Qwen3-8B"
LORA_PATH  = "./wechat_lora/checkpoint-25378"  # 改成你的最终checkpoint目录

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

# 一个尽量“稳”的生成配置（你可按需调整）
GEN_KW = dict(
    max_new_tokens=256,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.05
)

# 简易对话历史（防止越聊越长，做截断）
history = []
MAX_TURNS = 10  # 最多保留最近10轮

def build_prompt(history, user_text: str):
    # 纯 causal 拼接风格：用清晰分隔符降低“抢话/乱角色”
    # 如果你推理时有固定模板，也可以把这块替换成你自己的模板
    parts = []
    for u, a in history:
        parts.append(f"### User:\n{u}\n\n### Assistant:\n{a}\n")
    parts.append(f"### User:\n{user_text}\n\n### Assistant:\n")
    return "\n".join(parts)

print("\n=== Chat Debug Mode ===")
print("Commands: /reset  /exit  /show (show last prompt)\n")

last_prompt = None

while True:
    user_text = input("You> ").strip()
    if not user_text:
        continue
    if user_text == "/exit":
        break
    if user_text == "/reset":
        history = []
        print("System> history cleared.\n")
        continue

    prompt = build_prompt(history[-MAX_TURNS:], user_text)
    last_prompt = prompt

    if user_text == "/show":
        print("\n----- PROMPT -----")
        print(prompt)
        print("------------------\n")
        continue

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, **GEN_KW)

    text = tokenizer.decode(out[0], skip_special_tokens=True)

    # 只取最后一段 assistant 的新增部分
    ans = text.split("### Assistant:\n")[-1].strip()

    # 防止模型把下一轮 user 模板也生成出来
    if "### User:" in ans:
        ans = ans.split("### User:")[0].strip()

    print(f"Bot> {ans}\n")

    history.append((user_text, ans))