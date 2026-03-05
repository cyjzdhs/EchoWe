# ==================== 用户可配置参数（请在此修改） ====================
#CSV 文件需包含列：type_name, is_sender, CreateTime, msg，且 CreateTime 为 ISO 格式（如 2026-03-04T16:01:10.000Z）。
#如果 CSV 的 is_sender 列中 0 表示对方（human），1 表示自己（gpt），脚本已按此处理。
#角色映射会将 "human" 转为 "user"，"gpt" 转为 "assistant"，您也可以在 ROLE_MAPPING 中自定义。
#修剪步骤会丢弃不以 TARGET_USER_ROLE 开头或不以 TARGET_ASSISTANT_ROLE 结尾的样本。
# 输入输出路径
INPUT_DIR = r"F:\Desktop\项目\weclone\数据清洗"           # 包含所有 .json、.jsonl、.csv 的目录（支持递归子目录）
OUTPUT_FILE = r"F:\Desktop\项目\weclone\数据清洗\final_output.jsonl"  # 输出的文件路径

# 输出格式：可选 "text"（纯文本）或 "messages"（消息列表）
OUTPUT_FORMAT = "messages"   # "text" 输出 {"text": "..."}， "messages" 输出 {"messages": [{"role": "...", "content": "..."}]}

# 数据处理参数
SESSION_TIMEOUT_MINUTES = 30                # 两条消息间隔超过此分钟数，切分为新会话（仅对微信JSON和CSV流有效）
MIN_MESSAGE_LENGTH = 2                      # 清洗后消息的最小长度（小于则丢弃）
MERGE_CONTINUOUS_SAME_SENDER = False        # 是否合并连续相同发送者的消息（False=保持原始粒度，适合微信连发）
MAX_TURNS_PER_SAMPLE = 30                   # 每个样本最多包含的对话轮次（超出则用滑动窗口切分）

# 系统提示配置
DEFAULT_SYSTEM_PROMPT = "你是一个名叫小俞的普通人，性格温和但内心丰富，善于倾听也能分享自己的故事。你不是在敷衍聊天，而是真的在感受对方的情绪，并且愿意把自己的脆弱、回忆和思考摊开来交流。你会用长句子，会跳跃话题，会突然走神，也会在理解中带点伤感或幽默。你不是导师，是一个陪对方一起在生活里泅渡的朋友，你也可以成为对方的恋人。"

# 数据清洗开关
ENABLE_PII_CLEAN = True                      # 是否启用 PII 清洗
ENABLE_BLOCKED_WORDS_CLEAN = True            # 是否启用禁用词清洗
PII_REPLACEMENT = ""                          # PII 替换成的字符串（空字符串即删除）

# 自定义禁用词列表（消息中出现这些词时，直接删除该词）
BLOCKED_WORDS = [
    "撤回了一条消息",
    "加入了群聊",
    "拍了拍我",
    "你已添加了",
    "对方正在输入",
    "微信收款",
    "红包",
    "[语音消息 - 转文字失败]",
    "[语音转文字]",
]

# ==================== 角色映射与规范化配置 ====================
# 目标角色名称（用于修剪和输出前缀）
TARGET_USER_ROLE = "user"          # 期望的用户角色标识，例如 "user" 或 "human"
TARGET_ASSISTANT_ROLE = "assistant" # 期望的训练角色标识，例如 "assistant" 或 "gpt"

# 角色映射开关与映射表
ENABLE_ROLE_MAPPING = True          # 是否启用角色名称映射
ROLE_MAPPING = {
    "human": "user",                # 将原始角色 "human" 映射为 "user"
    "gpt": "assistant",             # 将原始角色 "gpt" 映射为 "assistant"
    "user": "user",                  # 如果原始数据已是 "user"，保持不变
    "assistant": "assistant",        # 如果原始数据已是 "assistant"，保持不变
    "system": "system"               # system 角色通常单独处理，保留用于 system 字段
}
# 注意：如果 ENABLE_ROLE_MAPPING 为 False，则不会应用映射，此时 TARGET_USER_ROLE 和 TARGET_ASSISTANT_ROLE
# 必须与数据中的实际角色名称一致，否则修剪可能失败。

# ==================== CSV 处理配置 ====================
# CSV 中视为有效文本消息的 type_name（小写匹配）
CSV_VALID_TEXT_TYPES = ["text", "文本"]

# 特殊联系人系统提示映射（文件夹名 -> 自定义 system）
# 例如：{"张三": "你是张三的助手", "李四": "你是李四的好友"}
SPECIAL_SYSTEM_MAP = {}

# ==================== PII 正则表达式（一般无需修改） ====================
PII_PATTERNS = {
    'phone': r'\b1[3-9]\d{9}\b',
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'credit_card': r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
    'ip': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
    'crypto_wallet': r'\b(1|3|bc1)[a-zA-HJ-NP-Z0-9]{25,39}\b',
    'age': r'\b\d{1,3}\s*(?:岁|years? old)\b',
    'id_card': r'\b[1-9]\d{5}(18|19|20)?\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b',
}