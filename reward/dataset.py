# dataset.py
import os
import json
import random
from torch.utils.data import Dataset

def load_pairwise_samples(data_dirs):
    """
    从多个目录中加载 output.json，并构造 pairwise 训练样本。
    
    对于 result 为 "PASS" 的条目：
      - 条目需要包含 "intent"、"prev_action" 和 "predict_action" 三个键；
      - 构造样本时，将上下文构造为：
            context = intent + "\n" + "\n".join(prev_action[:i+1])
        其中 i 的取值范围为从 1 到 min(len(prev_action), len(predict_action))-1，
        这样保证例如第二条预测时，上下文包含 intent 和 prev_action[0]、prev_action[1]，
        正候选预测为 predict_action[1]。
    
    对于 result 为 "FAIL" 的条目：
      - 将所有的 predict_action 动作加入负样本候选池。
    
    返回值：
       pairs: list of tuple (context, pos_predict, neg_predict)
    """
    pos_samples = []      # 存储 (context, pos_predict)
    neg_candidates = []   # 存储所有 FAIL 条目中的 predict_action 动作（字符串）

    for data_dir in data_dirs:
        json_path = os.path.join(data_dir, "output.json")
        if not os.path.exists(json_path):
            print(f"[Warning] File not found: {json_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            intent = item.get("intent", "").strip()
            prev_actions = item.get("prev_action", [])
            predict_actions = item.get("predict_action", [])
            result = item.get("result", "").upper()

            if result == "PASS":
                # 仅当有 intent 且 prev_actions 至少有1条（可以为 "None"）且 predict_actions 至少有2条时构造样本
                if intent and len(prev_actions) >= 1 and len(predict_actions) >= 2:
                    num_samples = min(len(prev_actions), len(predict_actions))
                    # 从 i==1 开始构造样本，此时上下文为：intent + prev_action[:i+1]
                    for i in range(1, num_samples):
                        context = intent + "\n" + "\n".join(prev_actions[:i+1])
                        pos_predict = predict_actions[i]
                        pos_samples.append((context, pos_predict))
            else:
                # 对于 FAIL 条目，将所有 predict_action 动作都加入负样本候选池
                for p in predict_actions:
                    neg_candidates.append(p)

    print(f"Loaded {len(pos_samples)} positive samples, {len(neg_candidates)} negative predict candidates.")
    # 打印一个负样本候选供检查
    if neg_candidates:
        print("Example negative candidate:", neg_candidates[0])
    else:
        print("No negative candidate found!")

    if not neg_candidates:
        raise ValueError("未能加载到任何失败项的预测候选，请检查数据！")

    pairs = []
    for context, pos_predict in pos_samples:
        neg_predict = random.choice(neg_candidates)
        pairs.append((context, pos_predict, neg_predict))
    return pairs

class PairwiseDataset(Dataset):
    def __init__(self, pairwise_samples, tokenizer, max_length=512):
        """
        参数：
           pairwise_samples: list of tuple (context, pos_predict, neg_predict)
           tokenizer: 分词器对象
           max_length: 最大序列长度
        """
        self.samples = pairwise_samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context, pos_predict, neg_predict = self.samples[idx]

        def format_input(context, candidate):
            return (
                "[Intent]\n"
                + context.split("\n", 1)[0].strip()  # 提取 intent
                + "\n\n[Previous Actions]\n"
                + context.split("\n", 1)[1].strip()  # 提取 prev_action 部分
                + "\n\n[Candidate Action]\n"
                + candidate
            )

        input_pos = format_input(context, pos_predict)
        input_neg = format_input(context, neg_predict)

        pos_encoding = self.tokenizer(
            input_pos,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        neg_encoding = self.tokenizer(
            input_neg,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids_pos": pos_encoding["input_ids"].squeeze(0),
            "attention_mask_pos": pos_encoding["attention_mask"].squeeze(0),
            "input_ids_neg": neg_encoding["input_ids"].squeeze(0),
            "attention_mask_neg": neg_encoding["attention_mask"].squeeze(0),
        }

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_dirs = [
        "../VisualWebArena/classifieds_gpt4v_som",
        "../VisualWebArena/reddit_gpt4v_som",
        "../VisualWebArena/shopping_gpt4v_som",
    ]
    pairwise_samples = load_pairwise_samples(data_dirs)
    dataset = PairwiseDataset(pairwise_samples, tokenizer, max_length=512)

    # 打印格式化输入文本
    sample = dataset[0]
    print("\n=== 正样本输入（input_ids_pos 解码） ===")
    print(tokenizer.decode(sample["input_ids_pos"], skip_special_tokens=True))

    print("\n=== 负样本输入（input_ids_neg 解码） ===")
    print(tokenizer.decode(sample["input_ids_neg"], skip_special_tokens=True))