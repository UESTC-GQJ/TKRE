
import pickle
import faiss
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
import numpy as np
import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils import set_seed, collate_fn, loadLogger, find_best_performance
from prepro import TACREDProcessor, RETACREDProcessor
from model import REModel, REModel_Repres
from torch.cuda.amp import GradScaler
import pdb


def obtain_repres(args, model, features):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)
    keys, repres = [], []
    for i_b, batch in enumerate(dataloader):
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'ss': batch[3].to(args.device),
                  'os': batch[4].to(args.device),
                  }
        # keys += batch[2].tolist()
        with torch.no_grad():
            repre = model(**inputs)
            
        # pdb.set_trace()
        repres += repre.tolist()

    # keys = np.array(keys, dtype=np.int64)
    repres = np.array(repres, dtype=np.int64)
    
    return repres


def faiss_build_index(train_representations, repre_size):
    index = faiss.IndexFlatL2(repre_size)
    print(index.is_trained)
    index.add(train_representations)
    print(index.ntotal)
    
    return index


def faiss_find_k_nearest(batch_reps, train_data, faiss_index, k):
    # 这里使用facebook的faiss包
    # batch_reps: batch * repre
    batch_reps = batch_reps.cpu().numpy()
    D, I = faiss_index.search(batch_reps, k) # xq为待检索向量，返回的I为每个待检索query最相似TopK的索引list，D为其对应的距离

    sorted_eucidean = torch.from_numpy(D)
    sorted_indices = torch.from_numpy(I)
    
    # pdb.set_trace()
    
    sorted_train_data = []
    for sample_idx in range(batch_reps.shape[0]):
        line_train_data = []
        line_index = sorted_indices[sample_idx]
        for item_index in line_index:
            line_train_data.append(train_data[item_index])
        sorted_train_data.append(line_train_data)
        # sorted_train_data.append(train_data[sorted_indices[sample_idx]])

    return sorted_eucidean, sorted_train_data


def knn_recall_sents(args, model, features, train_data, faiss_index, k):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)
    
    knn_aug_data = []
    for i_b, batch in enumerate(dataloader):
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'ss': batch[3].to(args.device),
                  'os': batch[4].to(args.device),
                  }
        with torch.no_grad():
            repre = model(**inputs)
        
        _, sorted_train_data = faiss_find_k_nearest(batch_reps=repre, train_data=train_data, faiss_index=faiss_index, k=k)
        knn_aug_data.extend(sorted_train_data)
    
    return knn_aug_data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data/tacred", type=str)
    parser.add_argument("--train_filename", default="train.json", type=str)
    parser.add_argument("--data_test_dir", default="./data/tacred", type=str)
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str)
    parser.add_argument("--input_format", default="typed_entity_marker_punct", type=str,
                        help="in [entity_mask, entity_marker, entity_marker_punct, typed_entity_marker, typed_entity_marker_punct]")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated.")

    parser.add_argument("--test_batch_size", default=32, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=42)
    
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    
    parser.add_argument("--checkpoint_filepath", type=str, default="")
    

    args = parser.parse_args()
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    if args.seed > 0:
        set_seed(args)

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    model = REModel_Repres(args, config)
    model.to(0)
    
    train_file = os.path.join(args.data_dir, args.train_filename)
    dev_file = os.path.join(args.data_dir, "dev.json")
    test_file = os.path.join(args.data_test_dir, "test.json")

    # TACREDProcessor: processing TACRED and TACREV datasets
    processor = TACREDProcessor(args, tokenizer)
    # RETACREDProcessor: processing ReTACRED datasets
    # processor = RETACREDProcessor(args, tokenizer)
    train_features = processor.read(train_file)
    dev_features = processor.read(dev_file)
    test_features = processor.read(test_file)
    
    with open(train_file, 'r') as f_in:
        train_data = json.load(f_in)

    if len(processor.new_tokens) > 0:
        model.encoder.resize_token_embeddings(len(tokenizer))
    
    train_model = torch.load(os.path.join(args.checkpoint_filepath, 'model.pt'))
    model.load_state_dict(train_model.state_dict())
    
    print('successfully load the model')
    
    train_repres = obtain_repres(args, model, train_features)
    
    faiss_index = faiss_build_index(train_repres, 2048)
    
    knn_aug_data = knn_recall_sents(args, model, test_features, train_data, faiss_index, k=8)
    
    with open(test_file, 'r') as f_in:
        test_data = json.load(f_in)
    
    assert len(test_data) == len(knn_aug_data)
    
    augmented_test_data = []
    for idx in range(len(test_data)):
        line = test_data[idx]
        line['knn_aug_data'] = knn_aug_data[idx]
        augmented_test_data.append(line)
    
    with open(os.path.join(args.checkpoint_filepath, 'tacred_augment_data.json'), 'w') as f_out:
        json.dump(augmented_test_data, f_out, indent=True)


if __name__ == '__main__':
    main()