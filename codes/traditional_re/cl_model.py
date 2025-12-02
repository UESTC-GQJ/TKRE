import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
import random
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from transformers import (
    RobertaTokenizer, 
    RobertaForMaskedLM, 
    PreTrainedModel,
    RobertaConfig,
    TrainingArguments, 
    Trainer
)
from transformers.modeling_outputs import MaskedLMOutput

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TKREOutput(MaskedLMOutput):
    """
    自定义模型输出，包含SCL loss
    """
    loss: Optional[torch.FloatTensor] = None
    mslm_loss: Optional[torch.FloatTensor] = None
    scl_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None

class TKREDataset(Dataset):
    """
    TKRE 数据集：同时支持 MSLM (Stage 1) 和 SCL (Stage 2) 的数据需求。
    Ref: [cite: 140, 169]
    """
    def __init__(self, file_path: str, tokenizer: RobertaTokenizer, relation2id: Dict[str, int], max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.relation2id = relation2id
        self.data = self.load_data(file_path)
        
    def load_data(self, file_path: str) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def find_token_span(self, input_ids: List[int], target_ids: List[int]) -> List[int]:
        """寻找子序列位置"""
        if not target_ids: return []
        n, m = len(input_ids), len(target_ids)
        for i in range(n - m + 1):
            if input_ids[i:i+m] == target_ids:
                return list(range(i, i+m))
        return []

    def get_span_masks(self, input_ids: List[int], subject: str, object: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成 MSLM 概率矩阵以及 SCL 所需的 Positive/Negative Span Masks。
        Ref: [cite: 61, 199]
        """
        seq_len = len(input_ids)
        
        mslm_probs = torch.full((seq_len,), 0.2) # Other tokens: 0.2 [cite: 186]
        pos_span_mask = torch.zeros(seq_len, dtype=torch.bool)
        neg_span_mask = torch.zeros(seq_len, dtype=torch.bool)
        
        # 特殊 Token 处理
        special_mask = self.tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
        mslm_probs.masked_fill_(torch.tensor(special_mask, dtype=torch.bool), 0.0)

        sub_tokens = self.tokenizer.encode(subject, add_special_tokens=False)
        obj_tokens = self.tokenizer.encode(object, add_special_tokens=False)
        sub_idx = self.find_token_span(input_ids, sub_tokens)
        obj_idx = self.find_token_span(input_ids, obj_tokens)

        # 实体区域 MSLM 概率设为 0.5 [cite: 186]
        if sub_idx: mslm_probs[sub_idx] = 0.5
        if obj_idx: mslm_probs[obj_idx] = 0.5

        # Ref: "positive spans (semantically aligned with target relations)" [cite: 61]
        # 定义为 Subject 和 Object 之间的区域
        if sub_idx and obj_idx:
            start = min(sub_idx[-1], obj_idx[-1]) + 1
            end = max(sub_idx[0], obj_idx[0]) - 1
            
            if start <= end:
                # MSLM 概率 0.8 [cite: 186]
                mslm_probs[start : end + 1] = 0.8
                # SCL Positive Span
                pos_span_mask[start : end + 1] = True
        
        # 4. 确定 Negative Span
        # Ref: "negative spans (relationally discordant but contextually plausible)" [cite: 61]
        # 策略：随机选择一个非实体、非关系区域的连续片段
        candidate_indices = [
            i for i in range(seq_len) 
            if not special_mask[i] 
            and i not in (sub_idx + obj_idx) 
            and not pos_span_mask[i]
        ]
        
        if candidate_indices:
            # 随机选取一个长度适中的片段作为负样本 (例如长度 1 到 3)
            neg_len = random.randint(1, min(3, len(candidate_indices)))
            start_pos = random.randint(0, len(candidate_indices) - neg_len)
            selected_neg_indices = candidate_indices[start_pos : start_pos + neg_len]
            neg_span_mask[selected_neg_indices] = True
            
        return mslm_probs, pos_span_mask, neg_span_mask

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['generated_text']
        
        # Tokenize
        inputs = self.tokenizer(
            text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        # 获取 Masks
        input_id_list = input_ids.tolist()
        mslm_probs, pos_mask, neg_mask = self.get_span_masks(input_id_list, item['subject'], item['object'])

        # 生成 MSLM Labels (公式 4) [cite: 190]
        masked_indices = torch.bernoulli(mslm_probs).bool() & attention_mask.bool()
        labels = input_ids.clone()
        labels[~masked_indices] = -100
        
        # 应用 Mask 到 Input
        input_ids_masked = input_ids.clone()
        input_ids_masked[masked_indices] = self.tokenizer.mask_token_id

        return {
            'input_ids': input_ids_masked,
            'attention_mask': attention_mask,
            'labels': labels,
            'pos_span_mask': pos_mask,      # SCL 输入
            'neg_span_mask': neg_mask,      # SCL 输入
            'relation_id': torch.tensor(self.relation2id.get(item['relation'], 0)) # Anchor ID
        }

    def __len__(self):
        return len(self.data)


class TKREModel(RobertaForMaskedLM):
    """
    TKRE 模型主体，集成 MSLM 和 SCL。
    继承自 RobertaForMaskedLM 以复用其 MLM head。
    Ref: [cite: 54, 66]
    """
    def __init__(self, config: RobertaConfig, num_relations: int, lambda1: float = 1.0, lambda2: float = 1.0, temperature: float = 0.1):
        super().__init__(config)
        self.num_relations = num_relations
        self.lambda1 = lambda1 # MSLM weight [cite: 210]
        self.lambda2 = lambda2 # SCL weight [cite: 210]
        self.temperature = temperature # Tau [cite: 204]
        
        # Anchor Embedding: 为每个关系类型学习一个锚点表示 h_a 
        self.relation_embeddings = nn.Embedding(num_relations, config.hidden_size)
        
        # SCL Projection Head (Optional but recommended for Contrastive Learning)
        self.scl_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

    def span_pooling(self, last_hidden_state: torch.Tensor, span_mask: torch.Tensor) -> torch.Tensor:
        """
        对 Span 内的 token embedding 进行平均池化。
        """
        # span_mask: [batch_size, seq_len]
        # hidden: [batch_size, seq_len, hidden_size]
        
        # 扩展 mask 维度以匹配 hidden state
        extended_mask = span_mask.unsqueeze(-1).float() # [B, L, 1]
        
        # 求和
        sum_embeddings = torch.sum(last_hidden_state * extended_mask, dim=1)
        
        # 计数 (避免除零)
        sum_mask = torch.clamp(extended_mask.sum(dim=1), min=1e-9)
        
        # 平均
        return sum_embeddings / sum_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        pos_span_mask=None,
        neg_span_mask=None,
        relation_id=None,
        **kwargs
    ):
        # 获取 embeddings 用于 SCL
        outputs = super().forward(
            input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            output_hidden_states=True, 
            return_dict=True,
            **kwargs
        )
        
        mslm_loss = outputs.loss # RobertaForMaskedLM 自动计算了 CrossEntropy
        
        total_loss = 0.0
        scl_loss_val = torch.tensor(0.0).to(input_ids.device)

        if mslm_loss is not None:
            total_loss += self.lambda1 * mslm_loss

        # 只有在提供了 mask 和 relation_id 时才计算 (Validation时可能不需要)
        if pos_span_mask is not None and neg_span_mask is not None and relation_id is not None:
            # 获取最后一层隐藏状态
            last_hidden_state = outputs.hidden_states[-1]
            
            # 获取 Anchor Embedding
            h_a = self.relation_embeddings(relation_id) # [batch, hidden]
            
            # 获取 Positive Span Embedding
            h_p = self.span_pooling(last_hidden_state, pos_span_mask) # [batch, hidden]
            
            # 获取 Negative Span Embedding
            h_n = self.span_pooling(last_hidden_state, neg_span_mask) # [batch, hidden]
            
            h_a = F.normalize(self.scl_head(h_a), p=2, dim=1)
            h_p = F.normalize(self.scl_head(h_p), p=2, dim=1)
            h_n = F.normalize(self.scl_head(h_n), p=2, dim=1)
            
            # 计算相似度 sim(h_a, h_x)
            sim_pos = torch.sum(h_a * h_p, dim=1) / self.temperature
            sim_neg = torch.sum(h_a * h_n, dim=1) / self.temperature
            
            logits = torch.stack([sim_pos, sim_neg], dim=1)
 
            scl_targets = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
            
            loss_fct = nn.CrossEntropyLoss()
            scl_loss_val = loss_fct(logits, scl_targets)
            
            total_loss += self.lambda2 * scl_loss_val

        return TKREOutput(
            loss=total_loss,
            mslm_loss=mslm_loss,
            scl_loss=scl_loss_val,
            logits=outputs.logits
        )

def train_tkre_stage2(data_path: str, output_dir: str):
    model_name = "roberta-large"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    relations = ["per:place_of_birth", "org:founded_by", "no_relation", "per:employee_of"]
    relation2id = {r: i for i, r in enumerate(relations)}

    dataset = TKREDataset(data_path, tokenizer, relation2id)

    config = RobertaConfig.from_pretrained(model_name)
    model = TKREModel.from_pretrained(
        model_name, 
        config=config,
        num_relations=len(relation2id),
        lambda1=1.0,    # MSLM 权重
        lambda2=1.0,    # SCL 权重
        temperature=0.1 # 温度系数
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        logging_steps=50,
        remove_unused_columns=False, # 关键：防止Trainer移除 pos_span_mask 等自定义列
        label_names=['labels', 'pos_span_mask', 'neg_span_mask', 'relation_id'] # 显式指定
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    logger.info("Starting TKRE Stage-2 (MSLM + SCL) Pre-training...")
    trainer.train()
    model.save_pretrained(output_dir)

if __name__ == "__main__":
    train_tkre_stage2(
        data_path="./datasets/synthetic_data.json", 
        output_dir="./output/tkre_stage2_final"
    )