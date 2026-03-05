#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
本脚本已修改为自动兼容两种输入格式：

ShareGPT 格式：每行包含 "conversations" 数组（带 from/value）。

纯文本流格式：每行包含 "text" 字符串（由 USER: 和 ASSISTANT: 拼接而成）。

脚本执行流程：

调用智谱 API 对对话质量评分（1-5 分），支持断点续传。

显示评分分布统计。

根据分数和可选规则筛选高质量对话。

移除评分字段，生成可直接用于训练的文件。
在评分阶段，自动识别数据格式：若存在 "text" 字段，则从中提取最后一轮用户消息和助手消息（通过正则解析 USER: 和 ASSISTANT: 行）；若存在 "conversations"，则沿用原逻辑。

评分文件保留原始所有字段，仅添加 "quality_score"。

筛选后文件移除评分字段，输出格式与输入格式一致（即纯文本流输入 → 纯文本流输出，ShareGPT 输入 → ShareGPT 输出）。

使用方法
安装依赖：

bash
pip install aiohttp tqdm
修改脚本开头的配置区域（至少填写 API_KEY 和输入文件路径）。

运行脚本：

bash
python data_pipeline.py
"""

import json
import asyncio
import aiohttp
import os
import re
from tqdm import tqdm
from collections import Counter
from typing import List, Dict, Optional, Tuple

# ==================== 配置区域（请按需修改）====================

# 文件路径
INPUT_FILE = "chat_sharegpt.jsonl"          # 原始数据文件（支持 text 或 conversations）
SCORED_FILE = "chat_with_scores.jsonl"       # 评分后输出文件
FILTERED_FILE = "chat_filtered.jsonl"        # 最终筛选后文件

# API 配置
API_KEY = "你的API密钥"                       # 必填
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MODEL = "glm-4.7"

# 评分参数
CONCURRENCY = 3                               # 并发请求数
REQUEST_DELAY = 0                              # 请求间隔（秒）
TEMPERATURE = 0
MAX_TOKENS = 5
RETRY_TIMES = 3

# 筛选规则
MIN_SCORE = 3                                  # 最低保留分数（>=此值）
USE_EXTRA_RULE_FOR_SCORE3 = False              # 是否对3分对话使用额外规则
GOOD_KEYWORDS = ["谢谢", "加油", "理解", "支持", "相信", "建议", "觉得", "人生", "其实", "但是", "因为"]

# ============================================================

def build_prompt(user_msg: str, assistant_msg: str) -> str:
    return f"""请对以下一段聊天记录进行质量评分（1-5分，整数），主要考察：
- 情感共鸣：回复是否真诚、有温度，能接住情绪。
- 智慧深度：是否包含有用的建议或人生哲理。
- 语言自然：是否像真人聊天，不生硬。
- 上下文连贯：是否与用户消息衔接自然。
- 整体吸引力：对话是否有趣或有启发性。

只输出一个整数分数，不要解释。

用户：{user_msg}
助手：{assistant_msg}

分数："""

def extract_last_round_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    从纯文本流中提取最后一轮用户和助手消息。
    文本格式示例：
        [系统提示]
        USER: 消息1
        ASSISTANT: 回复1
        USER: 消息2
        ASSISTANT: 回复2
    返回 (user_msg, assistant_msg)，若提取失败返回 (None, None)
    """
    lines = text.split('\n')
    last_user = None
    last_assistant = None
    user_index = -1
    assistant_index = -1

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("USER:"):
            last_user = line[5:].strip()
            user_index = i
        elif line.startswith("ASSISTANT:"):
            last_assistant = line[10:].strip()
            assistant_index = i

    # 必须存在最后一条用户和助手，且用户出现在助手之前（或同一轮）
    if last_user is not None and last_assistant is not None and user_index < assistant_index:
        return last_user, last_assistant
    return None, None

async def fetch_score(session, user_msg: str, assistant_msg: str, semaphore, retry: int = RETRY_TIMES) -> int:
    prompt = build_prompt(user_msg, assistant_msg)
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}
    url = f"{BASE_URL}/chat/completions"

    for attempt in range(retry):
        try:
            async with semaphore:
                if REQUEST_DELAY > 0:
                    await asyncio.sleep(REQUEST_DELAY)
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        content = result["choices"][0]["message"]["content"].strip()
                        match = re.search(r'\d+', content)
                        if match:
                            score = int(match.group())
                            if 1 <= score <= 5:
                                return score
                        return 3  # 解析失败默认给3分
                    else:
                        text = await resp.text()
                        print(f"HTTP {resp.status}: {text[:200]}")
                        if attempt < retry - 1:
                            await asyncio.sleep(2 ** attempt)  # 指数退避
        except Exception as e:
            print(f"请求异常: {e}")
            if attempt < retry - 1:
                await asyncio.sleep(2 ** attempt)
    return 3  # 全部失败默认3分

async def score_data():
    """评分阶段"""
    # 读取所有数据
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_lines = [line.strip() for line in f if line.strip()]
    total = len(all_lines)
    print(f"总数据量：{total} 条")

    # 检查已处理条数（断点续传）
    processed = 0
    if os.path.exists(SCORED_FILE):
        with open(SCORED_FILE, 'r', encoding='utf-8') as f:
            processed = sum(1 for _ in f)
        print(f"已处理 {processed} 条，将从第 {processed+1} 条继续")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for idx, line in enumerate(all_lines):
            if idx < processed:
                continue  # 跳过已处理的行
            data = json.loads(line)

            # 提取 user_msg 和 assistant_msg（根据数据格式自动识别）
            user_msg, assistant_msg = None, None
            if "text" in data:
                # 纯文本流格式
                user_msg, assistant_msg = extract_last_round_from_text(data["text"])
            elif "conversations" in data:
                # ShareGPT 格式
                conv = data.get("conversations", [])
                for msg in conv:
                    if msg.get("from") == "human":
                        user_msg = msg.get("value", "")
                    elif msg.get("from") == "gpt":
                        assistant_msg = msg.get("value", "")
            else:
                # 未知格式，视为无效
                pass

            if user_msg and assistant_msg:
                tasks.append((idx, data, user_msg, assistant_msg))
            else:
                # 无效对话直接给1分，立即写入
                data["quality_score"] = 1
                with open(SCORED_FILE, 'a', encoding='utf-8') as fout:
                    fout.write(json.dumps(data, ensure_ascii=False) + '\n')

        if tasks:
            print(f"需要处理的剩余条目：{len(tasks)}")
            async def process_one(idx, data, user_msg, assistant_msg):
                score = await fetch_score(session, user_msg, assistant_msg, semaphore)
                data["quality_score"] = score
                # 立即写入
                with open(SCORED_FILE, 'a', encoding='utf-8') as fout:
                    fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                return idx, score

            task_list = [process_one(idx, data, user_msg, assistant_msg) for idx, data, user_msg, assistant_msg in tasks]
            for f in tqdm(asyncio.as_completed(task_list), total=len(task_list), desc="评分中"):
                idx, score = await f
        else:
            print("所有数据已评分，无需处理。")

def filter_data():
    """筛选阶段：根据分数和规则保留数据，并移除评分字段"""
    print("\n开始筛选数据...")
    with open(SCORED_FILE, 'r', encoding='utf-8') as fin:
        all_lines = [line.strip() for line in fin if line.strip()]
    total = len(all_lines)
    kept = 0
    filtered = 0

    with open(FILTERED_FILE, 'w', encoding='utf-8') as fout:
        for line in all_lines:
            data = json.loads(line)
            score = data.get("quality_score", 1)
            # 移除评分字段（仅从内存中删除，不影响输出）
            if "quality_score" in data:
                del data["quality_score"]

            # 判断是否保留
            keep = False
            if score >= MIN_SCORE:
                if USE_EXTRA_RULE_FOR_SCORE3 and score == 3:
                    # 对3分应用额外规则：如果是 ShareGPT 格式，取助手最后一条回复；如果是纯文本流，需要从 text 中提取最后一条助手消息
                    if "conversations" in data:
                        assistant_msg = ""
                        for msg in data.get("conversations", []):
                            if msg.get("from") == "gpt":
                                assistant_msg = msg.get("value", "")
                    elif "text" in data:
                        # 从 text 中提取最后一条助手消息
                        _, assistant_msg = extract_last_round_from_text(data["text"])
                    else:
                        assistant_msg = ""

                    if len(assistant_msg) > 15 and any(kw in assistant_msg for kw in GOOD_KEYWORDS):
                        keep = True
                    else:
                        keep = False
                else:
                    keep = True

            if keep:
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                kept += 1
            else:
                filtered += 1

    print(f"筛选完成：总 {total} 条，保留 {kept} 条，过滤 {filtered} 条。")
    return kept, filtered

def show_stats():
    """显示评分分布统计"""
    if not os.path.exists(SCORED_FILE):
        print("评分文件不存在，无法统计。")
        return
    scores = []
    with open(SCORED_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            scores.append(data.get('quality_score', 0))
    counter = Counter(scores)
    total = len(scores)
    print("\n评分分布：")
    for score in sorted(counter.keys()):
        count = counter[score]
        percent = count / total * 100
        print(f"{score}分: {count} 条, 占比 {percent:.1f}%")
    print(f"总计评分数据: {total} 条")

def main():
    print("="*50)
    print("数据评分与筛选一体化工具（兼容纯文本流）")
    print("="*50)

    # 检查必要配置
    if API_KEY == "你的API密钥":
        print("错误：请先在脚本中填写你的API_KEY！")
        return

    # 1. 评分阶段
    print("\n[1] 开始评分阶段")
    asyncio.run(score_data())

    # 2. 统计阶段
    print("\n[2] 评分统计")
    show_stats()

    # 3. 筛选阶段
    print("\n[3] 开始筛选阶段")
    kept, filtered = filter_data()

    # 4. 最终结果
    print("\n" + "="*50)
    print(f"处理完成！最终数据保存至：{FILTERED_FILE}")
    print(f"保留条数：{kept}")
    print("="*50)

if __name__ == "__main__":
    main()