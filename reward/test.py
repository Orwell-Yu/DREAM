from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载 Qwen2.5-0.5B 模型和分词器
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

print(model)