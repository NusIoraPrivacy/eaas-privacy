from util.parameters import get_args
from util.utils import get_pretrained_model, get_ft_pretrained_model, get_finetuned_model
from util.globals import *

import torch
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
from datasets import load_dataset
import numpy as np


from huggingface_hub import login
login(token="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")

def save_ft_model(model, args):
    save_path = f"{args.denoise_model_dir}/finetune/{args.base_model}"
    model.save_pretrained(save_path) 

class SeqFtDataset(Dataset):
    def __init__(self, dataset_name, mode, tokenizer, sample_size, args):
        self.config = self._get_dataset_config(dataset_name)
        
        dataset = load_dataset(self.config['dataset'], self.config['subset'], split=mode) if self.config['subset'] else load_dataset(self.config['dataset'], split=mode)
        
        # 获取数据子集
        total_samples = len(dataset)
        selected_indices = np.random.choice(total_samples, sample_size, replace=False)
        sample_data = dataset.select(selected_indices)
        text = sample_data[self.config['key']]
        labels = sample_data[self.config['label']]
        
        # 对于每日对话数据集，它包含多个句子的对话，因此将其合并为单个文本
        if dataset_name == 'daily_dialog':
            text = [" ".join(t) for t in text]
        
        self.inputs = tokenizer(text, truncation=True, padding='max_length', max_length=args.token_length, return_tensors='pt')
        self.labels = torch.tensor(labels)
    
    def _get_dataset_config(self, dataset_name):
        if dataset_name not in ft_dataset_configs:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        return ft_dataset_configs[dataset_name]
        
    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs['input_ids'][idx],
            'attention_mask': self.inputs['attention_mask'][idx],
            'labels': self.labels[idx]
        }

if __name__ == "__main__":
    args = get_args()
    args.base_model = "bert-base-uncased"
    args.ft_data_size = 1000
    args.denoise_model_dir = "/home/hzyr/llm/denoise/model"
    tokenizer, base_model = get_ft_pretrained_model(args)
    optimizer = torch.optim.Adam(params =  base_model.parameters(), lr=1e-5)
    base_model = base_model.to(args.device)

    train_dataset = SeqFtDataset(args.ft_data, 'train', tokenizer, args.ft_data_size, args)
    dataloader = DataLoader(train_dataset, batch_size=args.ft_batch_size, shuffle=False, 
                            num_workers=0, drop_last=True, pin_memory=True)

    # finetune the bert model with classification dataset
    for i, batch in enumerate(tqdm(dataloader)):
        for key in batch:
            batch[key] = batch[key].to(args.device)
        output = base_model(**batch)
        optimizer.zero_grad()
        loss = output.loss
        loss.backward()
        optimizer.step()
    
    save_ft_model(base_model, args)