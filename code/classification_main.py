import json
import os
from pathlib import Path
import openai
from typing import List, Tuple
import csv


def read_single_json_file_and_concatenate_texts(file_path: str, max_chars: int = 1000) -> str:
    """
    读取单个json文件，提取text字段并拼接，控制总字符数为max_chars
    """
    all_texts = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取所有text字段
        for item in data:
            if isinstance(item, dict) and 'text' in item and item['text'].strip():
                all_texts.append(item['text'])

    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return ""

    # 用换行符拼接所有文本
    concatenated_text = '\n'.join(all_texts)

    # 控制字符总数为max_chars
    if len(concatenated_text) > max_chars:
        concatenated_text = concatenated_text[:max_chars]

    return concatenated_text


def get_all_json_files(base_dir: str) -> List[str]:
    """
    获取所有JSON文件的路径
    """
    # json_files = ['D:/zhuiyi/rag_chunk/20250908_LLM分段/20250908_LLM分段/data/测试数据/output.json']
    json_files = []

    # # 查找所有json文件
    json_dirs = [
        # os.path.join(base_dir, "data", "OCR效果性能测试_raw_process"),
        # os.path.join(base_dir, "data", "产研-文档问答效果评测_raw_process"),
        os.path.join(base_dir, "data", "测试数据")
    ]

    for json_dir in json_dirs:
        if os.path.exists(json_dir):
            for file_path in Path(json_dir).glob("*.json"):
                if str(file_path)[-19:] == 'json_processed.json':
                    json_files.append(str(file_path))

    return sorted(json_files)


def load_prompt_templates(prompt_dir: str) -> Tuple[str, str]:
    """
    读取system.txt和user.txt文件
    """
    system_file = os.path.join(prompt_dir, "doc_classification_system.txt")
    user_file = os.path.join(prompt_dir, "doc_classification_user.txt")

    with open(system_file, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    with open(user_file, 'r', encoding='utf-8') as f:
        user_template = f.read()

    return system_prompt, user_template


import requests
import json
import logging
from jinja2 import Template

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def call_llm(system_prompt: str, user_message: str, model: str) -> str:
    api_key = "zhuiyi"
    base_url = "http://172.18.160.39:8168/v1"
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=2470,
            top_p=0.8,
            presence_penalty=0,
            frequency_penalty=0,
            seed=1995,
            extra_body={"enable_thinking": False}
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"调用大模型时出错: {e}")
        return f"错误: {e}"


def main():
    # 设置工作目录
    base_dir = "D:/zhuiyi/rag_chunk/20250908_LLM分段/20250908_LLM分段"

    try:
        # 1. 获取所有JSON文件
        print("正在查找JSON文件...")
        json_files = get_all_json_files(base_dir)
        print(f"找到 {len(json_files)} 个JSON文件")

        # 2. 读取提示词模板
        print("正在读取提示词模板...")
        prompt_dir = os.path.join(base_dir, "prompt/classification_prompt")
        system_prompt, user_template = load_prompt_templates(prompt_dir)

        # 3. 对每个JSON文件分别处理
        for i, json_file in enumerate(json_files, 1):
            print(f"\n{'=' * 60}")
            print(f"正在处理第 {i} 个文件: {os.path.basename(json_file)}")
            print('=' * 60)

            # 读取并拼接单个文件中的文本
            concatenated_text = read_single_json_file_and_concatenate_texts(json_file, max_chars=1000)
            print(concatenated_text)
            print(f"拼接后的文本长度: {len(concatenated_text)} 字符")

            # 替换用户模板中的占位符
            user_message = user_template.replace("{{document_first_1000_character}}", concatenated_text)

            # 调用大模型
            print("正在调用大模型...")
            result = call_llm(system_prompt, user_message)

            # 输出结果
            print("-" * 40)
            print(result)
            print("-" * 40)

            # 添加短暂延迟，避免API调用过于频繁
            import time
            time.sleep(0.5)

        print(f"\n{'=' * 60}")
        print(f"处理完成！共处理了 {len(json_files)} 个文件")
        print('=' * 60)

    except Exception as e:
        print(f"程序执行出错: {e}")


if __name__ == "__main__":
    main()
