#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
遇到无法处理的把脚本和数据例子给ai然后生成一个适配的
数据处理主脚本（自动读取 config.py 中的配置）
处理多种格式的原始数据，生成训练用的 JSONL 文件。
支持格式：
- 微信导出 JSON（含 messages 数组）
- WeFlow 单文件 JSON（顶层包含消息数组）
- WeFlow JSONL（每行包含 _type 字段）
- ShareGPT JSONL（每行包含 conversations 数组）
- CSV（列：type_name, is_sender, CreateTime, msg）
输出格式可配置为纯文本（{"text": "..."}）或消息列表（{"messages": [{"role": "...", "content": "..."}]}）。
"""

import os
import re
import json
import glob
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

# 导入配置
try:
    import config
except ImportError:
    print("错误：找不到 config.py，请确保它与本脚本在同一目录下。")
    exit(1)

# 从config获取配置
INPUT_DIR = config.INPUT_DIR
OUTPUT_FILE = config.OUTPUT_FILE
OUTPUT_FORMAT = getattr(config, "OUTPUT_FORMAT", "messages")   # 默认 messages
SESSION_TIMEOUT_MINUTES = config.SESSION_TIMEOUT_MINUTES
MIN_MESSAGE_LENGTH = config.MIN_MESSAGE_LENGTH
MERGE_CONTINUOUS_SAME_SENDER = config.MERGE_CONTINUOUS_SAME_SENDER
MAX_TURNS_PER_SAMPLE = config.MAX_TURNS_PER_SAMPLE
DEFAULT_SYSTEM_PROMPT = config.DEFAULT_SYSTEM_PROMPT
ENABLE_PII_CLEAN = config.ENABLE_PII_CLEAN
ENABLE_BLOCKED_WORDS_CLEAN = config.ENABLE_BLOCKED_WORDS_CLEAN
PII_REPLACEMENT = config.PII_REPLACEMENT
BLOCKED_WORDS = config.BLOCKED_WORDS
PII_PATTERNS = config.PII_PATTERNS

# 角色映射相关
ENABLE_ROLE_MAPPING = config.ENABLE_ROLE_MAPPING
ROLE_MAPPING = config.ROLE_MAPPING
TARGET_USER_ROLE = config.TARGET_USER_ROLE
TARGET_ASSISTANT_ROLE = config.TARGET_ASSISTANT_ROLE

# CSV 相关
CSV_VALID_TEXT_TYPES = getattr(config, "CSV_VALID_TEXT_TYPES", ["text", "文本"])
SPECIAL_SYSTEM_MAP = getattr(config, "SPECIAL_SYSTEM_MAP", {})

# 编译PII正则
PII_REGEX = {name: re.compile(pattern) for name, pattern in PII_PATTERNS.items()}

# ==================== 清洗函数 ====================

def remove_control_chars(text: str) -> str:
    """移除除换行、制表符、回车外的控制字符"""
    if not isinstance(text, str):
        return text
    # 保留 \n (0x0A), \t (0x09), \r (0x0D)
    return re.sub(r'[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]', '', text)

def clean_pii(text: str) -> str:
    if not ENABLE_PII_CLEAN:
        return text
    for regex in PII_REGEX.values():
        text = regex.sub(PII_REPLACEMENT, text)
    return text

def clean_blocked_words(text: str) -> str:
    if not ENABLE_BLOCKED_WORDS_CLEAN:
        return text
    for word in BLOCKED_WORDS:
        text = text.replace(word, "")
    return text

def clean_message(text: str) -> str:
    text = remove_control_chars(text)
    text = clean_pii(text)
    text = clean_blocked_words(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_valid_message(text: str) -> bool:
    return len(text) >= MIN_MESSAGE_LENGTH

# ==================== 处理微信导出 JSON ====================

def parse_wechat_json_safe(json_path: str) -> Tuple[Optional[List[Dict]], bool]:
    """
    尝试用微信导出格式解析 JSON 文件。
    返回 (消息列表, 是否成功) 元组。
    - 成功且文件格式正确时返回 (消息列表, True)
    - 文件格式正确但无消息时返回 ([], True)
    - 解析失败时返回 (None, False)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return None, False

    messages = data.get('messages', [])
    if not messages:
        return [], True
    parsed = []
    for msg in messages:
        ts = msg.get('createTime')
        if not ts:
            continue
        try:
            ts = int(ts)
        except:
            continue
        is_sender = 1 if msg.get('isSend') == 1 else 0
        content = msg.get('content', '')
        if not content:
            continue
        parsed.append({
            'time': ts,
            'is_sender': is_sender,
            'msg': content
        })
    return parsed, True

# ==================== 处理 WeFlow 单文件 JSON ====================

def parse_weflow_single_json(json_path: str) -> List[Dict]:
    """
    解析 WeFlow 导出的单文件 JSON 格式
    返回消息列表（time, is_sender, msg）。
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  读取 WeFlow 单文件失败: {e}")
        return []

    # 检查是否为 WeFlow 格式（存在 weflow 或 session 键）
    if 'weflow' not in data and 'session' not in data:
        return []

    # 查找消息数组（顶层中值为列表且包含消息字段的键）
    messages = None
    for key, value in data.items():
        if isinstance(value, list) and len(value) > 0:
            first = value[0]
            if isinstance(first, dict) and 'createTime' in first and 'isSend' in first and 'content' in first:
                messages = value
                break
    if messages is None:
        return []

    parsed = []
    for msg in messages:
        ts = msg.get('createTime')
        if not ts:
            continue
        try:
            ts = int(ts)
        except:
            continue
        is_send = msg.get('isSend')
        if is_send is None:
            continue
        try:
            is_sender = int(is_send)
        except:
            continue
        if is_sender not in (0, 1):
            continue
        content = msg.get('content', '')
        if not content:
            continue
        # 按消息类型过滤（只保留文本消息）
        msg_type = msg.get('type', '').strip().lower()
        if msg_type not in CSV_VALID_TEXT_TYPES:
            continue

        parsed.append({
            'time': ts,
            'is_sender': is_sender,
            'msg': content
        })
    return parsed

# ==================== 处理 WeFlow JSONL ====================

def parse_weflow_jsonl(jsonl_path: str) -> List[Dict]:
    """
    解析 WeFlow 导出的 JSONL 文件（包含 _type 字段），返回消息列表（time, is_sender, msg）。
    需要从 header 和 member 中识别导出者 ID，以区分发送者。
    """
    messages = []
    exporter_name = None
    member_map = {}  # accountName -> platformId
    message_rows = []  # 暂存原始消息行

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except:
                    continue

                typ = obj.get('_type')
                if typ == 'header':
                    exporter_name = obj.get('meta', {}).get('name')
                elif typ == 'member':
                    account = obj.get('accountName')
                    pid = obj.get('platformId')
                    if account and pid:
                        member_map[account] = pid
                elif typ == 'message':
                    # 暂存消息，后续确定 exporter_id 后再处理
                    ts = obj.get('timestamp')
                    sender = obj.get('sender')
                    content = obj.get('content', '')
                    if ts and sender and content:
                        message_rows.append({
                            'time': int(ts),
                            'sender': sender,
                            'msg': content
                        })
    except Exception as e:
        print(f"  读取 WeFlow JSONL 失败: {e}")
        return []

    # 确定导出者 ID
    exporter_id = None
    if exporter_name and exporter_name in member_map:
        exporter_id = member_map[exporter_name]
    else:
        print(f"  警告：无法从 {jsonl_path} 中确定导出者 ID，所有消息将标记为对方发言")

    # 构建消息列表
    for row in message_rows:
        if exporter_id is not None:
            is_sender = 1 if row['sender'] == exporter_id else 0
        else:
            is_sender = 0  # 默认对方
        messages.append({
            'time': row['time'],
            'is_sender': is_sender,
            'msg': row['msg']
        })

    return messages

# ==================== 处理 CSV ====================

def parse_csv_file(csv_path: str) -> List[Dict]:
    """解析 CSV 文件，返回消息列表（time, is_sender, msg）"""
    messages = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 过滤消息类型
                type_name = row.get('type_name', '').strip().lower()
                if type_name not in CSV_VALID_TEXT_TYPES:
                    continue
                # 解析时间
                create_time_str = row.get('CreateTime', '')
                if not create_time_str:
                    continue
                try:
                    # 处理 ISO 格式（如 2026-03-04T16:01:10.000Z）
                    if create_time_str.endswith('Z'):
                        create_time_str = create_time_str.replace('Z', '+00:00')
                    dt = datetime.fromisoformat(create_time_str)
                    ts = int(dt.timestamp())
                except Exception:
                    continue
                # 发送者：0 为对方（human），1 为自己（gpt）
                try:
                    is_sender = int(row.get('is_sender', 0))
                except ValueError:
                    continue
                if is_sender not in (0, 1):
                    continue
                msg = row.get('msg', '')
                if not msg:
                    continue
                messages.append({
                    'time': ts,
                    'is_sender': is_sender,
                    'msg': msg
                })
    except Exception as e:
        print(f"读取 CSV 失败 {csv_path}: {e}")
    return messages

# ==================== 处理 ShareGPT JSONL ====================

def parse_sharegpt_jsonl(jsonl_path: str) -> List[Dict]:
    """
    解析 ShareGPT JSONL 文件，返回样本列表（直接使用 conversations 和 system）。
    同时清洗每条消息内容。兼容 from/value 和 role/content 两种字段名。
    """
    samples = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except:
                    continue
                conv = obj.get('conversations', [])
                system = obj.get('system', DEFAULT_SYSTEM_PROMPT)
                if not conv or not isinstance(conv, list):
                    continue
                # 清洗每条消息内容
                cleaned_conv = []
                for msg in conv:
                    if not isinstance(msg, dict):
                        continue
                    # 兼容两种字段名
                    role = msg.get("from") or msg.get("role")
                    content = msg.get("value") or msg.get("content")
                    if role is None or content is None:
                        continue
                    role = str(role)
                    cleaned = clean_message(content)
                    if not is_valid_message(cleaned):
                        continue
                    cleaned_conv.append({"from": role, "value": cleaned})
                if len(cleaned_conv) < 2:
                    continue
                samples.append({
                    "conversations": cleaned_conv,
                    "system": system
                })
    except Exception as e:
        print(f"读取 ShareGPT JSONL 失败 {jsonl_path}: {e}")
    return samples

# ==================== 通用会话处理函数 ====================

def split_sessions_by_time(messages: List[Dict], timeout_minutes: int) -> List[List[Dict]]:
    """按时间间隔将消息列表切分为多个会话"""
    if not messages:
        return []
    messages_sorted = sorted(messages, key=lambda x: x['time'])
    sessions = []
    current = [messages_sorted[0]]
    last_time = messages_sorted[0]['time']
    for msg in messages_sorted[1:]:
        if msg['time'] - last_time > timeout_minutes * 60:
            sessions.append(current)
            current = []
        current.append(msg)
        last_time = msg['time']
    if current:
        sessions.append(current)
    return sessions

def process_message_stream(messages: List[Dict], system_prompt: Optional[str] = None) -> List[Dict]:
    """
    处理消息流（来自微信 JSON、CSV 或 WeFlow）：
    - 按时间切分会话
    - 清洗每条消息
    - 过滤无效消息
    - 合并连续相同发送者（可选）
    - 滑动窗口生成对话样本
    返回样本列表，每个样本为 {"conversations": [{"from": role, "value": msg}], "system": system_prompt}
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    # 清洗消息
    cleaned_msgs = []
    for m in messages:
        cleaned = clean_message(m['msg'])
        if not is_valid_message(cleaned):
            continue
        cleaned_msgs.append({
            'time': m['time'],
            'is_sender': m['is_sender'],
            'msg': cleaned
        })
    if not cleaned_msgs:
        return []

    # 切分会话
    sessions = split_sessions_by_time(cleaned_msgs, SESSION_TIMEOUT_MINUTES)

    samples = []
    for sess in sessions:
        # 构建消息列表（未合并）
        conv = []
        for m in sess:
            role = "human" if m['is_sender'] == 0 else "gpt"
            conv.append({"from": role, "value": m['msg']})
        if not conv:
            continue

        # 合并连续相同发送者（可选）
        if MERGE_CONTINUOUS_SAME_SENDER:
            merged = []
            current = conv[0]
            for msg in conv[1:]:
                if msg["from"] == current["from"]:
                    current["value"] += "\n" + msg["value"]
                else:
                    merged.append(current)
                    current = msg
            merged.append(current)
            conv = merged

        # 滑动窗口生成样本（步长为2）
        for i in range(0, len(conv), 2):
            window = conv[i:i+MAX_TURNS_PER_SAMPLE]
            if len(window) < 2:
                break
            samples.append({
                "conversations": window,
                "system": system_prompt
            })
            if i + MAX_TURNS_PER_SAMPLE >= len(conv):
                break
    return samples

# ==================== 角色映射 ====================

def apply_role_mapping(sample: Dict) -> Optional[Dict]:
    """
    根据 ROLE_MAPPING 转换每条消息的角色名称。
    过滤掉未知角色或内容为空的轮次。
    """
    conv = sample.get('conversations', [])
    if not conv:
        return None
    new_conv = []
    for msg in conv:
        role = msg['from']
        if role not in ROLE_MAPPING:
            continue   # 跳过未知角色
        new_role = ROLE_MAPPING[role]
        # 内容可能为空（但之前已过滤，再次检查确保安全）
        if not msg['value'].strip():
            continue
        new_conv.append({"from": new_role, "value": msg['value']})
    if not new_conv:
        return None
    sample['conversations'] = new_conv
    return sample

# ==================== 修剪对话样本 ====================

def trim_conversation_sample(sample: Dict) -> Optional[Dict]:
    """
    修剪单个样本的对话，确保以 TARGET_USER_ROLE 开头、以 TARGET_ASSISTANT_ROLE 结尾。
    如果修剪后对话少于2轮，返回 None。
    """
    conv = sample.get('conversations', [])
    if not conv:
        return None

    # 去掉开头的非 TARGET_USER_ROLE
    start = 0
    while start < len(conv) and conv[start]['from'] != TARGET_USER_ROLE:
        start += 1
    conv = conv[start:]

    # 去掉结尾的非 TARGET_ASSISTANT_ROLE
    end = len(conv) - 1
    while end >= 0 and conv[end]['from'] != TARGET_ASSISTANT_ROLE:
        end -= 1
    conv = conv[:end+1]

    if len(conv) < 2:
        return None

    sample['conversations'] = conv
    return sample

# ==================== 转换为纯文本 ====================

def conversation_to_text(sample: Dict) -> str:
    """
    将单个样本转换为纯文本字符串（用于 OUTPUT_FORMAT="text"）。
    格式：system（如有）单独一行，然后每条消息为 "<角色名>: <内容>"
    """
    system = sample.get('system', '')
    conv = sample.get('conversations', [])
    lines = []
    if system:
        lines.append(system.strip())
    for msg in conv:
        # 直接使用消息中的角色名作为前缀（已通过角色映射标准化）
        lines.append(f"{msg['from']}: {msg['value']}")
    return "\n".join(lines)

# ==================== 主流程 ====================

def main():
    print("=" * 60)
    print("数据处理脚本（配置从 config.py 读取）")
    print("=" * 60)
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"输出格式: {OUTPUT_FORMAT}")
    print("-" * 60)

    # 收集所有文件（递归子目录）
    json_files = glob.glob(os.path.join(INPUT_DIR, "**", "*.json"), recursive=True)
    jsonl_files = glob.glob(os.path.join(INPUT_DIR, "**", "*.jsonl"), recursive=True)
    csv_files = glob.glob(os.path.join(INPUT_DIR, "**", "*.csv"), recursive=True)
    print(f"找到 {len(json_files)} 个 .json 文件，{len(jsonl_files)} 个 .jsonl 文件，{len(csv_files)} 个 .csv 文件")

    # 分别存储不同类型的样本/消息
    all_stream_messages = []          # 用于微信/CSV/WeFlow 消息流
    sharegpt_samples = []              # 用于 ShareGPT 格式样本

    # 处理 .json 文件（可能是微信导出、WeFlow 单文件、WeFlow JSONL、ShareGPT）
    for jf in json_files:
        # 1. 尝试微信导出格式
        msgs, success = parse_wechat_json_safe(jf)
        if success and msgs is not None:
            all_stream_messages.extend(msgs)
            print(f"  - 读取 {os.path.basename(jf)} (微信格式): {len(msgs)} 条原始消息")
            continue

        # 2. 尝试 WeFlow 单文件格式
        print(f"  尝试用 WeFlow 单文件格式解析 {os.path.basename(jf)}...")
        msgs = parse_weflow_single_json(jf)
        if msgs:
            all_stream_messages.extend(msgs)
            print(f"  - 读取 {os.path.basename(jf)} (WeFlow单文件): {len(msgs)} 条原始消息")
            continue

        # 3. 尝试 WeFlow JSONL 格式（多行）
        print(f"  尝试用 WeFlow JSONL 格式解析 {os.path.basename(jf)}...")
        msgs = parse_weflow_jsonl(jf)
        if msgs:
            all_stream_messages.extend(msgs)
            print(f"  - 读取 {os.path.basename(jf)} (WeFlow JSONL): {len(msgs)} 条原始消息")
            continue

        # 4. 尝试 ShareGPT 格式
        print(f"  尝试用 ShareGPT 格式解析 {os.path.basename(jf)}...")
        samples = parse_sharegpt_jsonl(jf)
        if samples:
            sharegpt_samples.extend(samples)
            print(f"  - 读取 {os.path.basename(jf)} (ShareGPT格式): {len(samples)} 个对话样本")
            continue

        print(f"  ⚠️ 无法解析 {os.path.basename(jf)}，已跳过")

    # 处理 .jsonl 文件（优先 ShareGPT，后备 WeFlow）
    for jlf in jsonl_files:
        # 先尝试 ShareGPT
        samples = parse_sharegpt_jsonl(jlf)
        if samples:
            sharegpt_samples.extend(samples)
            print(f"  - 读取 {os.path.basename(jlf)} (ShareGPT格式): {len(samples)} 个对话样本")
            continue

        # 再尝试 WeFlow JSONL
        msgs = parse_weflow_jsonl(jlf)
        if msgs:
            all_stream_messages.extend(msgs)
            print(f"  - 读取 {os.path.basename(jlf)} (WeFlow JSONL): {len(msgs)} 条原始消息")
            continue

        print(f"  ⚠️ 无法解析 {os.path.basename(jlf)}，已跳过")

    # 处理 CSV 文件
    for cf in csv_files:
        msgs = parse_csv_file(cf)
        if msgs:
            all_stream_messages.extend(msgs)
            print(f"  - 读取 {os.path.basename(cf)}: {len(msgs)} 条原始消息")
        else:
            print(f"  ⚠️ CSV 文件 {os.path.basename(cf)} 无有效消息，已跳过")

    # 将 all_stream_messages 转换为样本（消息流需要按联系人分组以应用不同的 system prompt）

    stream_samples = []
    if all_stream_messages:
        stream_samples = process_message_stream(all_stream_messages)  # 使用默认 system
        print(f"从消息流（微信/CSV/WeFlow）生成 {len(stream_samples)} 个对话样本")

    # 合并所有样本
    all_samples = stream_samples + sharegpt_samples
    print(f"总计获得 {len(all_samples)} 个对话样本")

    # 角色映射（如果启用）
    if ENABLE_ROLE_MAPPING:
        mapped_samples = []
        for s in all_samples:
            mapped = apply_role_mapping(s)
            if mapped:
                mapped_samples.append(mapped)
        print(f"角色映射后有效样本数: {len(mapped_samples)} (丢弃 {len(all_samples) - len(mapped_samples)} 个含未知角色的样本)")
        all_samples = mapped_samples

    # 修剪对话，确保每个样本以目标用户角色开头、目标助手角色结尾
    trimmed_samples = []
    for s in all_samples:
        trimmed = trim_conversation_sample(s)
        if trimmed:
            trimmed_samples.append(trimmed)
    print(f"修剪后有效样本数: {len(trimmed_samples)} (丢弃 {len(all_samples) - len(trimmed_samples)} 个不规范样本)")
    all_samples = trimmed_samples

    # 写入输出文件
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    total_output = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            if OUTPUT_FORMAT == "text":
                # 纯文本格式
                text = conversation_to_text(sample)
                if text.strip():
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
                    total_output += 1
            elif OUTPUT_FORMAT == "messages":
                # 消息列表格式
                conv = sample.get('conversations', [])
                if not conv:
                    continue
                messages = []
                # 可选项：将 system 作为第一条 system 消息（如需开启取消注释）
                # if sample.get('system'):
                #     messages.append({"role": "system", "content": sample['system']})
                for msg in conv:
                    role = msg['from']      # 已通过角色映射标准化为目标角色名
                    content = msg['value']
                    messages.append({"role": role, "content": content})
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + '\n')
                total_output += 1
            else:
                print(f"错误：未知的输出格式 {OUTPUT_FORMAT}")
                break

    print("-" * 60)
    print(f"处理完成！共生成 {total_output} 条样本，保存至 {OUTPUT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    main()