import re
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# ====== 路径改这里 ======
model_path = "/workspace/user-data/swift_finetune/models/DeepSeek-R1-0528-Qwen3-8B"
data_path  = "./wechat_stream.jsonl"  # 你的数据文件（jsonl）
# =======================

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

# LoRA（你可以保持一致）
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ---- 把 conversations 转成 token + labels（SFT response-only） ----
# 规则：human 段 labels=-100，gpt 段 labels=token_id
# 这样就不会训练模型去“预测用户说的话”

def build_tokens_and_labels(example):
    conv = example.get("conversations", [])
    sys = example.get("system", None)

    input_ids = []
    labels = []

    def add_text(text: str, supervise: bool):
        # supervise=False => labels=-100
        ids = tokenizer(text, add_special_tokens=False).input_ids
        input_ids.extend(ids)
        if supervise:
            labels.extend(ids)
        else:
            labels.extend([-100] * len(ids))

    # 可选 system：不监督（当作上下文）
    if sys:
        add_text(f"[SYSTEM]\n{sys}\n", supervise=False)

    # 严格按轮次处理
    for msg in conv:
        frm = msg.get("from")
        val = msg.get("value", "")
        if frm == "human":
            add_text(f"### User:\n{val}\n", supervise=False)
        elif frm == "gpt":
            add_text(f"### Assistant:\n{val}\n", supervise=True)
        else:
            # 未知角色：当作不监督上下文
            add_text(f"{val}\n", supervise=False)

    # 截断（保持 input_ids 和 labels 同步）
    max_len = 1536
    if len(input_ids) > max_len:
        input_ids = input_ids[-max_len:]
        labels    = labels[-max_len:]

    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

dataset = load_dataset("json", data_files=data_path)["train"]
dataset = dataset.map(build_tokens_and_labels, remove_columns=dataset.column_names, num_proc=4)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ====== 训练参数（SFT第二阶段建议更保守） ======
training_args = TrainingArguments(
    output_dir="./wechat_lora",                 # 你要求固定
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=1,                         # SFT建议先1 epoch
    learning_rate=2e-5,                         # 比你第一阶段更低
    bf16=True,
    logging_steps=20,
    save_steps=1000,
    save_total_limit=2,
    report_to="none",
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    group_by_length=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

trainer.train()