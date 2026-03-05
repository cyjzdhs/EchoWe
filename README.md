
# EchoWe

## 一、背景与目标
- 模型为 DeepSeek-R1-0528-Qwen3-8B（Qwen3 架构），使用 LoRA 高效微调。
- 训练方式为因果语言建模（Causal LM），不采用传统 SFT 配对，保留连发、跳跃等类似微信风格。
- 脚本都集成一个代码里面，必须仔细阅读代码注释或者说明，防止出错
- 原理：标准 Causal LM 方式训练（prompt + response 都算 loss），而不是像 SFT 那样只对 response 算 loss。但是为了对齐对话，会再次sft训练
## 二、环境要求
# 可以试试lab4ai(送50代金券够完成了) https://www.lab4ai.cn/register?promoteID=user-8NddZi47ik
- **GPU**：H800 80G（或类似显存，可根据显存调整 batch size 和 max_length）
- **Python**：3.10 或 3.11（避免 3.12 的兼容问题）
- **PyTorch**：≥ 2.1（推荐 2.7.1 与 CUDA 12.8 组合）
- **Transformers**：推荐 4.56.0（与 PyTorch 2.7.1 兼容）
- **PEFT**：≥ 0.10.0
- **其他依赖**：datasets, accelerate, bitsandbytes, matplotlib（用于绘图）

### 2.1 创建 Conda 环境（Python 3.11）
```bash
conda create -n wechat python=3.11 -y
conda activate wechat
```

### 2.2 安装 PyTorch（CUDA 12.8 版本，适配 H800）
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

### 2.3 安装训练依赖
```bash
pip install transformers==4.56.0 peft==0.12.0 datasets accelerate bitsandbytes numpy==1.26.4 matplotlib
```

### 2.4 验证安装
```bash
python -c "import torch, transformers, numpy; print('torch:', torch.__version__); print('transformers:', transformers.__version__); print('numpy:', numpy.__version__); print('CUDA available:', torch.cuda.is_available())"
```
预期输出：
```
torch: 2.7.1
transformers: 4.56.0
numpy: 1.26.4
CUDA available: True
```

## 三、数据准备
WeFlow导出来适配json和weclon的cvs
### 3.1 原始数据格式要求
支持两种输入格式：
- **导出 JSON**：文件应包含 `messages` 数组，每条消息有 `createTime`（整数时间戳）、`isSend`（1=发送，0=接收）、`content`（文本内容）。
- **ShareGPT JSONL**：每行一个 JSON，包含 `"conversations"` 数组，元素为 `{"from": "human"/"gpt", "value": "..."}`，可选的 `"system"` 字段。
- **weclone cvs**：CSV 文件需包含列：type_name, is_sender, CreateTime, msg，且 CreateTime 为 ISO 格式（如 2026-03-04T16:01:10.000Z）。
如果 CSV 的 is_sender 列中 0 表示对方（human），1 表示自己（gpt），脚本已按此处理。
角色映射会将 "human" 转为 "user"，"gpt" 转为 "assistant"，您也可以在 ROLE_MAPPING 中自定义。
修剪步骤会丢弃不以 TARGET_USER_ROLE 开头或不以 TARGET_ASSISTANT_ROLE 结尾的样本。
将所有原始文件（`.cvs``.json` 和 `.jsonl`）放入同一个目录（例如 `raw_data/`）。

### 3.2 数据清洗与转换脚本
提供两个脚本：
- **`config.py`**：配置文件，用于设置路径和参数。字段不一样必须清洗。
- **`process.py`**：数据处理主脚本，读取配置，执行清洗和转换。

#### 3.2.1 配置文件 `config.py`
```bash
# 根据实际路径修改 INPUT_DIR 和 OUTPUT_FILE
# 其他参数可根据需要调整
```

#### 3.2.2 处理脚本 `process.py`
与 `config.py` 放在同一目录。运行前确保 `config.py` 中路径已修改。
```bash
python process.py
```
脚本会遍历输入目录中的所有 `.cvs``.json` 和 `.jsonl` 文件，进行清洗、切分、合并，最终生成一个纯文本流 JSONL 文件（每行 `{"text": "..."}`），并输出统计信息。也支持导出json格式用于正常的stf训练，在cnfig脚本里面修改

**注意**：这里的清洗方式和weclone不同。本项目构造的是多轮对话，而非简单 QA 问答。微信聊天本身断断续续，因此采用的清洗逻辑更为激进。你可以根据实际需求调整清洗参数。同时也支持导出json格式用于正常的stf训练。

### 3.3 数据筛选与增强（可选）
本流程支持两种数据构造方式：
- **大合集**：直接使用全部数据训练，让模型学习整体风格。
- **精选数据**：调用 API 对对话质量打分（1-5 分），然后筛选高分数据，再调用 API 模拟人添加 `<think>` 思考标签，与高分数据混合进行二次微调。相关脚本（data_pipeline.py`）位于项目目录，可根据需要修改使用。不想调用就本地的打分，但是没有脚本，我没有弄。data_pipeline.py 目前仅支持 JSONL 格式的输入（每行一个样本，且必须包含 "text" 或 "conversations" 字段）


## 四、训练脚本

保存为 `train_final.py`和`train_sft.py`。脚本已优化为支持命令行参数 `--output_dir`。


## 五、启动训练

### 5.1 手动使用 tmux
```bash
# 创建新会话
tmux new -s wechat

# 在会话中运行训练
python train_final.py

# 按下 Ctrl+B 然后 D 脱离会话（训练继续后台运行）

# 重新进入会话
tmux attach -t wechat

# 查看所有会话
tmux ls
```

### 5.2 一键训练（推荐）
本系统提供自动化版本管理、后台运行、日志记录、GPU监控和训练报告生成。

#### 5.2.1 创建总控脚本 `train_manager.sh`
mkdir -p output/logs
mkdir -p output/reports
#### 5.2.2 赋予执行权限
```bash
chmod +x train_manager.sh
```

#### 5.2.3 启动训练
```bash
./train_manager.sh
```
loss_history.csv：所有记录的 loss 数据

summary.txt：训练摘要（总步数、最终 loss、平均 loss）

loss_curve.png：loss 曲线（如果安装了 matplotlib）

训练stf只需要
chmod +x train_sft_manager.sh
./train_sft_manager.sh
#### 5.2.4 查看结果
所有东西都在 output/
实时查看日志：tail -f output/logs/train_0302_0544.log

监控GPU：watch -n 1 nvidia-smi

如果需要终止训练，可以使用pkill -f train_final.py（

检查是否有旧进程残留，可以再次运行 pgrep -f train_final.py 确认只有当前这一个主进程（及其子进程）。

## 六、训练监控与判断

### 6.1 正常训练输出示例
- loss 在 0.8~1.2 之间波动是健康的。
- 梯度范数一般在 0.3~1.5 之间。
- 如果 loss 持续低于 0.8 且波动很小，可能开始过拟合，可提前停止（但跑满 2 epoch 通常安全）。

### 6.2 常见警告
- `Unrecognized keys in rope_scaling ...`：模型配置文件冗余字段，不影响训练，可忽略。
- `torch_dtype is deprecated`：提示使用 `dtype` 参数，不影响运行。

## 七、常见错误及解决方法

| 错误现象 | 可能原因 | 解决方法 |
|--------|--------|--------|
| `ValueError: expected sequence of length X at dim 1 (got Y)` | 自己手动构造了 `labels` 但未正确 padding，与 collator 冲突 | 使用 `DataCollatorForLanguageModeling` 自动生成 labels，不要在 tokenize 函数中手动添加 `labels` |
| `CUDA out of memory` | 显存不足 | 降低 `max_length`（例如 1024）或 `per_device_train_batch_size`（改为 1），或减少 `gradient_accumulation_steps` |
| `ImportError: flash_attn not installed` | 启用了 `attn_implementation="flash_attention_2"` 但未安装 flash-attn | 安装 `pip install flash-attn --no-build-isolation`，或去掉该参数使用默认 attention |
| `NameError: name 'tokenizer' is not defined` | 变量作用域错误 | 检查 tokenize 函数是否在 tokenizer 定义之后，或使用全局变量 |
| 数据加载慢 | `num_proc` 设置过低 | 适当增加 `num_proc`（不超过 CPU 核心数） |

## 八、训练后模型测试

训练完成后，LoRA 权重保存在输出目录中（例如 `wechat_lora_时间戳`，包含 `adapter_model.safetensors` 和 `adapter_config.json`）。推理时需加载 base 模型和 LoRA 权重。

创建 `inference.py`：

运行：
```bash
python inference.py
```

## 九、最佳实践与注意事项

1. **不要手动构造 labels**：使用 `DataCollatorForLanguageModeling` 自动处理 labels 和 padding，避免长度不一致错误。
2. **max_length 设置**：类似于的微信聊天实际长度多在几百 token，1536 已足够，若显存不足可降至 1024。
3. **epoch 数**：2 epoch 通常足够，过多易过拟合（loss 持续低于 0.8 且对话变得机械）。
4. **学习率**：5e-5 对 LoRA 是常用值，若收敛过快可适当降低（如 2e-5）。
5. **思考标签处理**：训练时保留 `<think>` 可以让模型学会内部推理；若想最终不输出，可在 tokenize 前用正则去除：
   ```python
   import re
   example["text"] = re.sub(r"<think>.*?</think>", "", example["text"], flags=re.DOTALL)
   ```
6. **挂后台**：务必使用 tmux 或 `nohup` + `train_manager.sh` 防止 SSH 断开导致训练中断。
7. **数据清洗**：若不需要清洗 PII，可将 `ENABLE_PII_CLEAN` 设为 `False`；禁用词列表可根据实际需要增删。清洗只是删除这个片段不会整句话删除。在启用角色映射（ENABLE_ROLE_MAPPING=True）时，确保 ROLE_MAPPING 覆盖所有可能出现的角色，否则未知角色的轮次会被丢弃。
8. 我的想法，Qwen3 本身支持多模态，可通过 `[图片]` 符号或 AI 生成的聊天数据（含 `<think>`）来适应聊天中的图片场景。无需单独训练图片，保留原有图片理解能力即可，所以没有适配多模态的训练

## 十、继续微调

如需基于已有 LoRA 继续训练，修改训练脚本，使用 `PeftModel.from_pretrained` 加载已有权重，并适当降低学习率（如 1e-5），再训练 1 epoch。

示例：
```python
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(...)
model = PeftModel.from_pretrained(model, "wechat_lora_v1")
# 继续训练...
```

## 十一、总结


本流程构建了一个完整的人格模型训练体系，包含环境配置、数据准备、训练、监控、测试和继续微调。关键在于数据构造方式符合类似于微信聊天的自然流，而非简单 QA 对。建议在实际使用中根据数据特点灵活调整清洗参数和训练策略。标准 Causal LM 方式训练（prompt + response 都算 loss），而不是像 SFT 那样只对 response 算 loss。目的在于先注入风格在进行对话，因为大部分聊天数据都是跳跃很大的。然后记得读注释，修改基础配置，不然就丢给ai读再教你。




