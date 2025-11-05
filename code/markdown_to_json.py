import json
from classification_main import call_llm, get_all_json_files
from template import other_user_prompt, other_system_prompt
from jinja2 import Template
from split_chunk import make_text


def markdown_to_json_lines(markdown_content):
    """
    将Markdown内容按行提取为指定JSON格式
    """
    lines = markdown_content.split('\n')
    result = []
    index = 1
    for i, line in enumerate(lines, 1):
        if line != "":
            result.append({
                "index": index,
                "text": line
            })
            index += 1

    return result


def view_markdown(file_path):
    """查看Markdown文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            print("=" * 50)
            print(f"文件: {file_path}")
            print("=" * 50)
            print(content)
            print("=" * 50)
            print(f"总行数: {len(content.splitlines())}")
        return content
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return ''
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return ''


# 使用示例
# markdown_content = view_markdown('../data/markdown_data/guangxijianzhu.md')

with open("../prompt/make_markdown_prompt/system.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

with open("../prompt/make_markdown_prompt/user.txt", "r", encoding="utf-8") as f:
    user_prompt = f.read()

base_dir = "D:/zhuiyi/rag_chunk/20250908_LLM分段/20250908_LLM分段"

# 1. 获取所有JSON文件
print("正在查找JSON文件...")
json_files = get_all_json_files(base_dir)

import csv

with open('../result_csv/markdown_output.csv', 'w', encoding='utf-8') as f:
    fieldnames = ['name', 'chunk_index', 'text', 'score', 'result']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for i, json_file in enumerate(json_files, 1):
        name_start = json_file.rfind("\\")
        name = json_file[name_start+1:]
        if name != '中国人寿保险说明书-5336.pdf.json_processed.json':
            continue
        else:
            pass
        print(name, "正在markdown风格化")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)[:175]
            chunk_result, text = make_text(str([[1, len(data)]]), data, len(data))

        user_template = Template(other_user_prompt)
        with open("../prompt/make_markdown_prompt/user.txt", "r", encoding="utf-8") as f:
            user_prompt = f.read()
        user_prompt = user_prompt.replace('{{txt_text}}', text)

        result = call_llm(system_prompt, user_prompt, model="Qwen3-32B")

        markdown_result_start = result.find('```markdown')
        markdown_result_end = result.rfind('```')
        markdown_result = result[markdown_result_start + 12:markdown_result_end]

        # 转换并保存为JSON
        json_result = markdown_to_json_lines(markdown_result)

        # 保存到文件
        with open('../data/markdown_data/'+name, 'w', encoding='utf-8') as json_data:
            json.dump(json_result, json_data, ensure_ascii=False, indent=2)

        user_content = user_template.render(name_json=name, content_json=json_result)

        print("正在分块")
        chunk = call_llm(other_system_prompt, user_content, model="Qwen3-32B")
        print(chunk)

        chunk_result, text = make_text(chunk, json_result, total_data_len=len(chunk_result))

        file_record = {'name': name, 'chunk_index': str(chunk_result), 'text': text}
        writer.writerow(file_record)
