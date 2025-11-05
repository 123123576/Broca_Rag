import json, argparse

def convert_one(old):
    """
    old: {"text": "1：我想把套餐……", "label": ["降档意图"]}
    new: {"messages": [system, user, assistant]}
    """
    label = label = "|".join(old["label"]) if old["label"] else "无"                # 使用逗号拼接标签
    text  = old["text"]
    prompt = """
    你是对话文本分类助手。请从以下8个候选标签中，选择所有符合对话内容的标签（单选或多选），多个标签用英文竖线"|"分隔，输出时禁止添加任何额外字符。若对话内容与所有标签均无关，请输出"无"。

    候选标签：
    1. 携号转网（用户明确表达更换运营商意向）
    2. 携转挽留（客服尝试挽留携转用户）
    3. 降档意图（用户要求降低套餐档位）
    4. 降档挽留（客服挽留降档用户）
    5. 宽带销户（用户要求注销宽带）
    6. 宽带销户挽留（客服挽留宽带销户）
    7. 销户确认（用户确认完成销户，无需挽留）
    8. 销户挽留（客服挽留一般性销户）

    特殊规则：
    - 当用户与客服出现标签冲突时，以用户意图为准（如用户说"我要销户"客服说"不建议"，应选"销户确认"）
    - 禁止输出标签以外的任何解释
    """
    new = {
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user",   "content": f"文本：{text}"},
            {"role": "assistant", "content": label}
        ]
    }
    return new

def main(in_file, out_file):
    with open(in_file, encoding="utf-8") as f:
        data = json.load(f)          # 如果是 jsonl 请改用 json.loads(line)
    new_data = [convert_one(d) for d in data]
    with open(out_file, "w", encoding="utf-8") as f:
        for d in new_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"已转换 {len(new_data)} 条，保存为 {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file",  default="train.json")
    parser.add_argument("--out_file", default="train.jsonl")
    args = parser.parse_args()
    main(args.in_file, args.out_file)