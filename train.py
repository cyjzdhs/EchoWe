import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# =========================
# GPU 加速
# =========================

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =========================
# 路径
# =========================

model_path = "/workspace/user-data/swift_finetune/models/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
data_path = "/workspace/user-data/swift_finetune/wechat_stream.jsonl"

# =========================
# Tokenizer
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================
# 模型
# =========================

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

model.config.use_cache = False  # 训练必须关

# 开启 gradient checkpointing
model.gradient_checkpointing_enable()

# =========================
# LoRA（降 rank 提速）
# =========================

lora_config = LoraConfig(
    r=16,                     # 从32降到16（提速）
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# =========================
# 数据
# =========================

dataset = load_dataset("json", data_files=data_path)["train"]

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=1536
    )

dataset = dataset.map(
    tokenize,
    remove_columns=["text"],
    num_proc=4
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# =========================
# 训练参数（减少累积加速）
# =========================

training_args = TrainingArguments(
    output_dir = "./wechat_lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    learning_rate=5e-5,
    bf16=True,
    logging_steps=20,
    save_steps=1000,
    save_total_limit=2,
    report_to="none",
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    group_by_length=True
)

# =========================
# Trainer
# =========================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

trainer.train()