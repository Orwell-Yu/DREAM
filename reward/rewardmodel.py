import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class RewardModel(nn.Module):
    def __init__(self, base_model_name):
        super(RewardModel, self).__init__()
        # 加载预训练因果语言模型
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        # 添加 config 属性，供 PEFT/LoRA 使用
        self.config = self.base_model.config
        # 添加一个简单的值头，用于输出单一标量得分；模仿 LLaVA 的设计，将 bias 设置为 False
        self.value_head = nn.Linear(self.base_model.config.hidden_size, 1, bias=False)
        # 如果原模型有 prepare_inputs_for_generation 方法，则使用它
        if hasattr(self.base_model, "prepare_inputs_for_generation"):
            self.prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
        else:
            self.prepare_inputs_for_generation = lambda input_ids, **kwargs: {"input_ids": input_ids}

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # 请求返回所有隐藏状态
            return_dict=True
        )
        hidden_states = outputs.hidden_states[-1]
        seq_lengths = attention_mask.sum(dim=1) - 1  # [batch]
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_indices, seq_lengths, :]
        rewards = self.value_head(last_hidden).squeeze(-1)
        return rewards

    def save_pretrained(self, directory):
        """
        保存整个 RewardModel 的 state_dict，包括基础模型和自定义层，
        同时保存基础模型的 config 以供加载时参考。
        """
        os.makedirs(directory, exist_ok=True)
        # 保存整个模型的参数
        full_state_dict = self.state_dict()
        torch.save(full_state_dict, os.path.join(directory, "pytorch_model.bin"))
        # 保存基础模型配置（你也可以扩展保存额外的配置信息）
        self.config.to_json_file(os.path.join(directory, "config.json"))