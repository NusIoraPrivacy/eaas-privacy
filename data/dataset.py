import h5py
import torch
from torch.utils.data import Dataset, ConcatDataset
from util.utils import text2text_priv
from data.load_data import get_dataset_cpu, get_dataset_universe, get_dataset_sep
import numpy as np
from datasets import load_dataset
from util.globals import *
import copy

class HDF5Dataset(Dataset):
    def __init__(self, filepath):
        self.filepath = filepath
        with h5py.File(self.filepath, 'r') as f:
            self.length = len(f['input_ids'])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with h5py.File(self.filepath, 'r') as f:
            input_ids = torch.tensor(f['input_ids'][index])
            attention_masks = torch.tensor(f['attention_masks'][index])
            noises = torch.tensor(f['noises'][index])
            clean_cls_emb = torch.tensor(f['clean_cls_emb'][index])
            noise_cls_emb = torch.tensor(f['noise_cls_emb'][index])

        return input_ids, attention_masks, noises, clean_cls_emb, noise_cls_emb

class IMDBDataset(Dataset):
    def __init__(self, mode, tokenizer, args):
        self.inputs, self.labels = get_dataset_cpu('imdb', mode, tokenizer, args)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs['input_ids'][idx],
            'attention_mask': self.inputs['attention_mask'][idx],
            'label': self.labels[idx]
        }

class DownStreamDataset(Dataset):
    def __init__(self, mode, tokenizer, args):
        if mode == "train":
            inputs, labels, _, _ = get_dataset_universe(args.task, tokenizer, args, mode=mode)
            sample_length = args.downstream_task_train_size
        elif mode == "test":
            _, _, inputs, labels = get_dataset_universe(args.task, tokenizer, args, mode=mode)
            sample_length = args.downstream_task_test_size
        # print("sample_length = ", sample_length)
        self.n_keys = len(inputs)  # 直接从inputs列表长度获取n_keys
        selected_indices = np.random.choice(len(labels), min(sample_length, len(labels)), replace=False)
        
        self.labels = [labels[i] for i in selected_indices]
        self.labels = torch.stack(self.labels)
        
        self.inputs = []
        for idx in selected_indices:
            input_dict = {}
            for key_idx in range(self.n_keys):
                input_dict[f'input_ids{key_idx + 1}'] = inputs[key_idx]['input_ids'][idx]
                input_dict[f'attention_mask{key_idx + 1}'] = inputs[key_idx]['attention_mask'][idx]
            self.inputs.append(input_dict)
        
        print(len(self.inputs))
    

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_dict = self.inputs[idx]
        return {
            'input_ids': [torch.tensor(input_dict[f'input_ids{i + 1}']) for i in range(self.n_keys)],
            'attention_mask': [torch.tensor(input_dict[f'attention_mask{i + 1}']) for i in range(self.n_keys)],
            'labels': self.labels[idx]
        }

class DownStreamDatasetSep(Dataset):
    def __init__(self, mode, tokenizer, model, args):
        if mode == "train":
            inputs, labels, _, _ = get_dataset_sep(args.task, tokenizer, args, mode=mode)
            sample_length = args.downstream_task_train_size
        elif mode == "test":
            _, _, inputs, labels = get_dataset_sep(args.task, tokenizer, args, mode=mode)
            sample_length = args.downstream_task_test_size
        # print("sample_length = ", sample_length)
        selected_indices = np.random.choice(len(labels), min(sample_length, len(labels)), replace=False)
        
        self.labels = [labels[i] for i in selected_indices]
        self.labels = torch.stack(self.labels)
        
        raw_inputs = {key: torch.stack([inputs[key][i] for i in selected_indices]) for key in inputs.keys()}
        raw_inputs_copy = copy.deepcopy(raw_inputs)
        if args.test_eta != 0:
            self.priv_inputs = text2text_priv(raw_inputs_copy, tokenizer, model, args)
        else:
            self.priv_inputs = raw_inputs_copy
        self.plain_token = raw_inputs['input_ids'][:, :args.n_plain_tok]
        self.n_labels = len(torch.unique(self.labels))
        self.raw_inputs = raw_inputs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.priv_inputs['input_ids'][idx],
            'attention_mask': self.priv_inputs['attention_mask'][idx],
            'plain_token': self.plain_token[idx],
            'labels': self.labels[idx]
        }

class SyntheticDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs['input_ids'][idx],
            'attention_mask': self.inputs['attention_mask'][idx]
        }

class SingleDataset(Dataset):
    def __init__(self, config, tokenizer, token_length):
        dataset = load_dataset(config['dataset'], config['subset'], split=config['mode']) if config['subset'] else load_dataset(config['dataset'], split=config['mode'])
        total_samples = len(dataset)
        selected_indices = np.random.choice(total_samples, min(config['sample_size'], total_samples), replace=False)
        sample_dataset = dataset.select(selected_indices)
        text = sample_dataset[config['key']]
        
        if config['key'] == 'dialog':
            text = [" ".join(t) for t in text]
            
        self.inputs = tokenizer(text, truncation=True, padding='max_length', max_length=token_length, return_tensors='pt')

    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs['input_ids'][idx],
            'attention_mask': self.inputs['attention_mask'][idx]
        }

def create_mixed_dataset(config_list, tokenizer, token_length):
    configs = []
    for entry in config_list.split(', '):
        parts= entry.split(':')
        if len(parts) != 3:
            raise ValueError(f"Invalid entry format for '{entry}'. Expected format: 'name:mode:sample_size'")
        name, mode, sample_size=parts
        config = dataset_configs[name]
        config['mode'] = mode
        config['sample_size'] = int(sample_size)
        configs.append(config)

    datasets = [SingleDataset(config, tokenizer, token_length) for config in configs]
    return ConcatDataset(datasets)

class UniversalDataset(Dataset):
    def __init__(self, dataset_name, mode, tokenizer, size, args):
        self.config = self._get_dataset_config(dataset_name)
        
        dataset = load_dataset(self.config['dataset'], self.config['subset'], split=mode) if self.config['subset'] else load_dataset(self.config['dataset'], split=mode)
        
        # 获取数据子集
        total_samples = len(dataset)
        selected_indices = np.random.choice(total_samples, min(size, total_samples), replace=False)
        sample_dataset = dataset.select(selected_indices)
        text = sample_dataset[self.config['key']]
        
        # 对于每日对话数据集，它包含多个句子的对话，因此将其合并为单个文本
        if dataset_name == 'daily_dialog':
            text = [" ".join(t) for t in text]
        
        self.inputs = tokenizer(text, truncation=True, padding='max_length', max_length=args.token_length, return_tensors='pt')
    
    def _get_dataset_config(self, dataset_name):
        if dataset_name not in dataset_configs:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        return dataset_configs[dataset_name]
        
    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs['input_ids'][idx],
            'attention_mask': self.inputs['attention_mask'][idx]
        }