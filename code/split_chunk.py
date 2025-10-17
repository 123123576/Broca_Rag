from classification_main import call_llm, get_all_json_files, read_single_json_file_and_concatenate_texts
from classification_main import load_prompt_templates
import os
from typing import Tuple
import json
import csv
from template import other_system_prompt, other_user_prompt
from jinja2 import Template


def other_load_prompt_templates(prompt_dir: str) -> Tuple[str, str]:
    """
    读取system.txt和user.txt文件
    """
    system_file = os.path.join(prompt_dir, "system_v4.txt")
    user_file = os.path.join(prompt_dir, "user.txt")

    with open(system_file, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    with open(user_file, 'r', encoding='utf-8') as f:
        user_template = f.read()

    return system_prompt, user_template


def load_QA_prompt_templates(prompt_dir: str) -> Tuple[str, str]:
    """
    读取system.txt和user.txt文件
    """
    system_file = os.path.join(prompt_dir, "QA_segmentation_system_v1.txt")
    user_file = os.path.join(prompt_dir, "QA_segmentation_user_v4.txt")

    with open(system_file, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    with open(user_file, 'r', encoding='utf-8') as f:
        user_template = f.read()

    return system_prompt, user_template


def evaluate_load_prompt_templates(prompt_dir: str) -> Tuple[str, str]:
    """
    读取evaluate_system.txt和user.txt文件
    """
    system_file = os.path.join(prompt_dir, "evaluate_system_prompt.txt")
    user_file = os.path.join(prompt_dir, "evaluate_user_prompt.txt")

    with open(system_file, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    with open(user_file, 'r', encoding='utf-8') as f:
        user_template = f.read()

    return system_prompt, user_template


def make_text(chunk_result, data, total_data_len):
    """
    修复后的make_text函数，增加边界检查和错误处理
    """
    text = ''
    list_data = []

    # 检查chunk_result是否有效
    if not chunk_result or not isinstance(chunk_result, str):
        return list_data, text

    try:
        # 清理和验证JSON格式
        chunk_result = chunk_result.strip()
        if chunk_result.startswith('[') and len(chunk_result) > 2:
            # 修复不完整的JSON
            chunk_start = chunk_result.find('[[')
            if not chunk_result.endswith(']]'):
                chunk_end = chunk_result.rfind(']')
                if chunk_end > 0:
                    chunk_result = chunk_result[chunk_start:chunk_end + 1]
                    chunk_result += ']'
            # 解析JSON
            list_data = json.loads(chunk_result)

            # 处理每个分块
            for chunk in list_data:
                if not isinstance(chunk, list) or len(chunk) < 2:
                    continue

                start_idx = max(0, chunk[0] - 1)  # 确保索引不小于0
                end_idx = min(len(data), chunk[1])  # 确保索引不超过数据长度

                # 检查索引有效性
                if start_idx >= len(data) or start_idx > end_idx:
                    continue

                # 拼接文本
                for chunk_index in range(start_idx, end_idx):
                    if chunk_index < len(data):
                        text += data[chunk_index].get('text', '') + '\n'
                text += '\n\n'

    except (json.JSONDecodeError, IndexError, KeyError) as e:
        print(f"解析chunk_result时出错: {e}")
        print(f"原始chunk_result: {chunk_result}")
        list_data = []
        text = ''

    return list_data, text


def safe_call_llm(system_prompt, user_message, max_retries=3):
    """
    安全的LLM调用函数，包含重试机制
    """
    for attempt in range(max_retries):
        try:
            result = call_llm(system_prompt, user_message, "Bowen_General_v2.2_14B_20241001")
            if 'Error code:' not in result:
                return result
            print(f"LLM调用失败，第{attempt + 1}次重试...")
        except Exception as e:
            print(f"LLM调用异常: {e}, 第{attempt + 1}次重试...")

        import time
        time.sleep(2)  # 重试前等待2秒

    return "Error: Max retries exceeded"


def process_large_data(other_system_prompt, other_user_template, file_name, data, chunk_size=402):
    """
    安全地处理大数据的分块函数
    """
    chunk_result = []
    text = ''
    start_index = 0

    while start_index < len(data):
        # 计算结束索引，确保不越界
        end_index = min(start_index + chunk_size, len(data))
        some_data = data[start_index:end_index]
        try:
            # 准备用户消息
            user_template = Template(other_user_template)

            user_content = user_template.render(name_json=file_name,
                                                content_json=some_data)

            print('*'*60)
            print(user_content)
            # 安全调用LLM
            some_chunk_result = safe_call_llm(other_system_prompt, user_content)
            print(some_chunk_result)
            # 处理返回结果
            if 'Error code: 400' in some_chunk_result or 'Error:' in some_chunk_result:
                print(f"分块处理出错，跳过该分块: {some_chunk_result}")
                break

            some_list_data, text1 = make_text(some_chunk_result, some_data, len(data))

            # 调整索引偏移量
            for chunk in some_list_data:
                if len(chunk) >= 2:
                    # 将相对索引转换为绝对索引
                    adjusted_chunk = [chunk[0], chunk[1]]
                    chunk_result.append(adjusted_chunk)

            text += text1

            # 安全地更新start_index
            if some_list_data:
                last_chunk = some_list_data[-1]
                if len(last_chunk) == 2:
                    start_index = last_chunk[1]  # 使用绝对索引

            # 防止无限循环
            if start_index >= len(data) - 1:
                break

        except Exception as e:
            print(f"处理分块时出错: {e}")
            start_index = end_index  # 出错时移动到下一个分块

    return chunk_result, text


def main():
    base_dir = "D:/zhuiyi/rag_chunk/20250908_LLM分段/20250908_LLM分段"

    try:
        # 1. 获取所有JSON文件
        print("正在查找JSON文件...")
        json_files = get_all_json_files(base_dir)
        print(f"找到 {len(json_files)} 个JSON文件")

        # 2. 读取提示词模板
        print("正在读取提示词模板...")
        prompt_dir = os.path.join(base_dir, "prompt/classification_prompt")
        QA_prompt_dir = os.path.join(base_dir, "prompt/QA_segmentation_prompt")
        other_prompt_dir = os.path.join(base_dir, 'prompt/Other_segmentation_prompt')
        evaluate_prompt_dir = os.path.join(base_dir, 'prompt/evaluate_prompt')

        system_prompt, user_template = load_prompt_templates(prompt_dir)
        QA_system_prompt, QA_user_template = load_QA_prompt_templates(QA_prompt_dir)
        # other_system_prompt, other_user_template = other_load_prompt_templates(other_prompt_dir)
        evaluate_system_prompt, evaluate_template = evaluate_load_prompt_templates(evaluate_prompt_dir)
        QA_number = 0
        other_number = 0
        error_files = []

        with open('../result_csv/output_test_bowen1001_14B_v1.csv', 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['name', 'chunk_index', 'text', 'score', 'result']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # 3. 对每个JSON文件分别处理
            for i, json_file in enumerate(json_files, 1):
                try:
                    print(f"\n{'=' * 60}")
                    print(f"正在处理第 {i} 个文件: {os.path.basename(json_file)}")
                    print('=' * 60)

                    # 读取并拼接单个文件中的文本
                    concatenated_text = read_single_json_file_and_concatenate_texts(json_file, max_chars=1000)
                    print(f"拼接后的文本长度: {len(concatenated_text)} 字符")

                    # 获取json数据
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 替换用户模板中的占位符
                    user_message = user_template.replace("{{document_first_1000_character}}", concatenated_text)

                    # 安全调用大模型
                    print("正在调用大模型进行分类...")
                    classification_result = safe_call_llm(system_prompt, user_message)

                    print("-" * 40)
                    print(f"分类结果: {classification_result}")
                    print("-" * 40)

                    import time
                    time.sleep(0.5)

                    start = json_file.rfind("\\")
                    file_name = json_file[start + 1:]

                    if classification_result == '["other"]':
                        print("使用Other分段策略...")
                        chunk_result, text = process_large_data(other_system_prompt, other_user_prompt, file_name,
                                                                data)
                        other_number += 1
                    else:
                        print("使用QA分段策略...")
                        # 替换用户模板中的占位符
                        QA_user_message = QA_user_template.replace("{{replace_json}}", str(data))
                        chunk_result = safe_call_llm(QA_system_prompt, QA_user_message)

                        QA_number += 1
                        if 'Error code: 400' in chunk_result or 'Error:' in chunk_result:
                            print("QA分段出错，使用空结果")
                            chunk_result, text = [], ''
                        else:
                            chunk_result, text = make_text(chunk_result, data, total_data_len=len(data))

                    print(f"分块结果: {chunk_result}")

                    # 模型评估
                    print("正在评估模型分块结果")
                    evaluate_message = evaluate_template.replace("{{data}}", text)
                    result = call_llm(evaluate_system_prompt, evaluate_message, "Qwen3-235B-A22B-Instruct-2507")
                    import re
                    result_end = result.rfind("}")
                    result_start = result.find("{")
                    match = result[result_start:result_end+1]

                    print("评估结果:", result)
                    json_result = json.loads(match)
                    score = json_result['overall_assessment']['average_score']
                    score_result = json_result['overall_assessment']['summary']
                    # 写入CSV
                    file_record = {'name': file_name, 'chunk_index': str(chunk_result), 'text': text, 'score': str(score), 'result': str(score_result)}
                    writer.writerow(file_record)
                    print(f"文件 {file_name} 处理完成")

                except Exception as e:
                    error_msg = f"处理文件 {json_file} 时出错: {e}"
                    print(error_msg)
                    error_files.append(json_file)
                    continue

        print(f"\n{'=' * 60}")
        print(f"处理完成！共处理了 {len(json_files)} 个文件")
        print(f"QA分段文件: {QA_number} 个")
        print(f"Other分段文件: {other_number} 个")
        if error_files:
            print(f"出错文件 {len(error_files)} 个:")
            for ef in error_files:
                print(f"  - {ef}")
        print('=' * 60)

    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
