import json


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
markdown_content = view_markdown('../data/markdown_data/guangxijianzhu.md')

# 转换并保存为JSON
json_result = markdown_to_json_lines(markdown_content)

# 打印结果
print(json.dumps(json_result, ensure_ascii=False, indent=2))

# 保存到文件
with open('output.json', 'w', encoding='utf-8') as f:
    json.dump(json_result, f, ensure_ascii=False, indent=2)