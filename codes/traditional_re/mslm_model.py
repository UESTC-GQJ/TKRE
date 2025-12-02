import torch
from torch.utils.data import Dataset
import json
import logging
from typing import Dict, List, Tuple
from transformers import (
    RobertaTokenizer, 
    RobertaForMaskedLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RelationMLMDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: RobertaTokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(file_path)
        logger.info(f"Loaded {len(self.data)} samples from {file_path}")

    def load_data(self, file_path: str) -> List[Dict]:
        """加载JSON数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def find_token_span(self, input_ids: List[int], target_ids: List[int]) -> List[int]:
        """
        在input_ids中查找target_ids的起始和结束位置。
        返回所有属于target的索引列表。
        """
        if not target_ids:
            return []
            
        span_indices = []
        n = len(input_ids)
        m = len(target_ids)
        
        # 简单的滑动窗口搜索
        for i in range(n - m + 1):
            if input_ids[i:i+m] == target_ids:
                span_indices.extend(list(range(i, i+m)))
                break 
        return span_indices

    def get_span_mask_probabilities(self, input_ids: List[int], subject_text: str, object_text: str) -> torch.Tensor:
        """
        根据论文公式 (3) 生成每个Token的掩码概率矩阵。
        Ref: 
        - Relation spans: 0.8
        - Entity spans: 0.5
        - Other tokens: 0.2
        """
        probs = torch.full((len(input_ids),), 0.2)
        
        # 对特殊token (<s>, </s>, <pad>) 设置概率为 0，永远不mask
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
        ]
        probs.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool).squeeze(), 0.0)

        # 获取实体对应的 token ids (不包含特殊字符)
        sub_tokens = self.tokenizer.encode(subject_text, add_special_tokens=False)
        obj_tokens = self.tokenizer.encode(object_text, add_special_tokens=False)

        sub_indices = self.find_token_span(input_ids, sub_tokens)
        obj_indices = self.find_token_span(input_ids, obj_tokens)

        # 设置实体概率为 0.5
        if sub_indices:
            probs[sub_indices] = 0.5
        if obj_indices:
            probs[obj_indices] = 0.5

        if sub_indices and obj_indices:
            sub_start, sub_end = sub_indices[0], sub_indices[-1]
            obj_start, obj_end = obj_indices[0], obj_indices[-1]
            
            start_idx = min(sub_end, obj_end) + 1
            end_idx = max(sub_start, obj_start) - 1
            
            if start_idx <= end_idx:
                # 设置关系区间概率为 0.8
                probs[start_idx : end_idx + 1] = 0.8

        return probs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['generated_text']
        subject = item['subject']
        object = item['object']

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].squeeze(0)  # Shape: [seq_len]
        attention_mask = inputs['attention_mask'].squeeze(0)
        input_id_list = input_ids.tolist()
        probability_matrix = self.get_span_mask_probabilities(input_id_list, subject, object)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices = masked_indices & attention_mask.bool()

        labels = input_ids.clone()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        input_ids[masked_indices] = self.tokenizer.mask_token_id

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def train_relation_mlm(data_path: str, output_dir: str):

    model_name = "roberta-large" # 本地路径
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForMaskedLM.from_pretrained(model_name)
    
    dataset = RelationMLMDataset(data_path, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=5, # 根据数据量调整
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=50,
        fp16=torch.cuda.is_available(), # 如果有GPU则开启
        remove_unused_columns=False, # 防止Trainer自动删除我们自定义的Dataset列
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    # 5. 开始预训练
    logger.info("Starting MSLM Pre-training...")
    trainer.train()
    
    # 6. 保存
    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    # 请确保路径正确
    train_relation_mlm(
        data_path="./datasets/explanation_data.json", # 假设这是你的 explanation-driven corpus
        output_dir="./output/tkre_mslm_checkpoint"
    )