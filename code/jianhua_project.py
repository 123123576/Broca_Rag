import json
import os


def process_QA_json_file(input_file_path, output_file_path):
    """
    读取JSON文件，删除'style'字段，重新设置'index'字段为从0开始递增的值

    Args:
        input_file_path (str): 输入JSON文件路径
        output_file_path (str): 输出JSON文件路径
    """
    try:
        # 读取JSON文件
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        print(f"成功读取JSON文件，共{len(data)}条记录")

        # 处理每个条目
        processed_data = []
        for i, item in enumerate(data):
            # 创建新的条目，先设置index字段确保它在第一个位置
            new_item = {"index": str(i)}

            # 复制所有字段，除了'style'字段和'index'字段（因为已经设置了新的index）
            for key, value in item.items():
                if key != 'style' and key != 'index':
                    new_item[key] = value

            processed_data.append(new_item)

        # 保存处理后的数据
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(processed_data, file, ensure_ascii=False, indent=4)

        print(f"处理完成！结果已保存到: {output_file_path}")
        print(f"处理了{len(processed_data)}条记录")

        # 显示一些统计信息
        style_removed_count = 0
        for item in data:
            if 'style' in item:
                style_removed_count += 1

        if style_removed_count > 0:
            print(f"删除了{style_removed_count}个'style'字段")
        else:
            print("原数据中没有发现'style'字段")

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file_path}")
    except json.JSONDecodeError:
        print(f"错误：{input_file_path} 不是有效的JSON文件")
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")


def process_other_json_file(input_file_path, output_file_path):
    """
    读取JSON文件，保留'style','text'字段，重新设置'index'字段为从0开始递增的值

    Args:
        input_file_path (str): 输入JSON文件路径
        output_file_path (str): 输出JSON文件路径
    """
    try:
        # 读取JSON文件
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        print(f"成功读取JSON文件，共{len(data)}条记录")

        # 处理每个条目
        processed_data = []
        index = 0
        for i, item in enumerate(data):
            # 复制所有字段，除了'style'字段和'index'字段（因为已经设置了新的index）
            for key, value in item.items():
                if key == 'text' and value != '':
                    index += 1
                    # 创建新的条目，先设置index字段确保它在第一个位置
                    new_item = {"index": str(index)}
                    new_item['message'] = {
                        "messages": [{"role": "system", "content": value}, {"role": "assistant", "content": "无"}]}
                    processed_data.append(new_item)
                if key == "label":
                    new_item = {"label": str(value)}
                    processed_data.append(new_item)

        # 保存处理后的数据
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(processed_data, file, ensure_ascii=False, indent=4)

        print(f"处理完成！结果已保存到: {output_file_path}")
        print(f"处理了{len(processed_data)}条记录")

        # 显示一些统计信息
        style_removed_count = 0
        for item in data:
            if 'style' in item:
                style_removed_count += 1

        print(f"一共有{style_removed_count}个'style'字段")

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file_path}")
    except json.JSONDecodeError:
        print(f"错误：{input_file_path} 不是有效的JSON文件")
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")


def main():
    input_file = "../{}".format('raw_data_1029.json')
    output_file = "../{}_processed.json".format("raw_data_1029.json")

    print("开始处理JSON文件...")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print("-" * 50)

    # 处理文件
    process_other_json_file(input_file, output_file)


if __name__ == "__main__":
    main()
