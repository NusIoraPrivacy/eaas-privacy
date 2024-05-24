import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from datasets import load_dataset
import torch
import numpy as np
from transformers import BertForMaskedLM, Trainer, TrainingArguments
from torch.distributions.gamma import Gamma
from util.globals import *

def get_dataset(data_name, mode, tokenizer, args):
    '''
    mode: 'test' or 'train'
    '''
    if type(data_name) is str:
        dataset = load_dataset(data_name)
    else:
        dataset = data_name
    # load data
    text = dataset[mode]['text']
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    inputs = {name: tensor.to(args.device) for name, tensor in inputs.items()}
    labels = dataset[mode]['label']
    labels = torch.tensor(labels).to(args.device)
    return inputs, labels

def get_dataset_cpu(data_name, mode, tokenizer, args):
    '''
    mode: 'test' or 'train'
    '''
    dataset = load_dataset(data_name)
    # load data
    text = dataset[mode]['text']
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    inputs = {name: tensor for name, tensor in inputs.items()}
    labels = dataset[mode]['label']
    labels = torch.tensor(labels)
    return inputs, labels

def synthesize_data(size, token_length, tokenizer, args):
    # sample token and attention mask
    sim_lengths = torch.randint(1, token_length, size=(size,1)) # size = size
    vocab_size = tokenizer.vocab_size  #vocab_size = len(tokenizer)
    sim_tokens = torch.randint(0, vocab_size, size=(size, token_length))
    att_mask = torch.ones_like(sim_tokens)
    for i in range(size):
        this_len = sim_lengths[i]
        sim_tokens[i, this_len:] = 0
        att_mask[i, this_len:] = 0
    
    return {'input_ids': sim_tokens, 'attention_mask': att_mask}

def get_imdb_subset(tokenizer, device, sample_size=500):
    inputs, labels = get_dataset("imdb", 'train', tokenizer, device)
    subset_indices = np.random.choice(len(inputs['input_ids']), sample_size, replace=False)
    return {name: tensor[subset_indices] for name, tensor in inputs.items()}

def sample_noise_Gaussian(d_shape, noise_stddev, device="cpu"):
    noise = torch.normal(mean=0., std=noise_stddev, size=d_shape, device=device)
    return noise

def sample_noise_Chi(d_shape, eta, device="cpu"):
    n_dim = d_shape[-1]
    alpha = torch.ones(d_shape) * n_dim
    beta = torch.ones(d_shape) * eta
    m = Gamma(alpha, beta)
    l_lst = m.sample()
    v_lst = -2 * torch.rand(d_shape) + 1
    noise = l_lst * v_lst
    noise = noise.to(device)
    return noise

def get_dataset_universe(d_name, tokenizer, args, mode="all"):
    if d_name not in task_dataset_configs:
        raise ValueError(f"Unknown dataset name: {d_name}")

    config = task_dataset_configs[d_name]
    # get train dataset
    train_inputs, train_labels, test_inputs, test_labels = None, None, None, None
    if mode == "all" or mode == "train":
        train_dataset = load_dataset(config['dataset'], config['subset'], split=config['train']) if config['subset'] else load_dataset(config['dataset'], split=config['train'])
        train_inputs = []
        for key in config['key']:
            train_text = train_dataset[key]
            this_train_inputs = tokenizer(train_text, truncation=True, padding='max_length', max_length=args.token_length, return_tensors='pt')
            this_train_inputs = {name: tensor.to(args.device) for name, tensor in this_train_inputs.items()}
            train_inputs.append(this_train_inputs)
        train_labels = train_dataset['label']
        train_labels = torch.tensor(train_labels).to(args.device)
    # get test dataset
    if mode == "all" or mode == "test":
        test_dataset = load_dataset(config['dataset'], config['subset'], split=config['test']) if config['subset'] else load_dataset(config['dataset'], split=config['test'])
        test_inputs = []
        for key in config['key']:
            test_text = test_dataset[key]
            this_test_inputs = tokenizer(test_text, truncation=True, padding='max_length', max_length=args.token_length, return_tensors='pt')
            this_test_inputs = {name: tensor.to(args.device) for name, tensor in this_test_inputs.items()}
            test_inputs.append(this_test_inputs)
        test_labels = test_dataset['label']
        test_labels = torch.tensor(test_labels).to(args.device)
    return train_inputs, train_labels, test_inputs, test_labels

def load_dataset_sep(config, tokenizer, args, mode):
    dataset = load_dataset(config['dataset'], config['subset'], split=config[mode]) if config['subset'] else load_dataset(config['dataset'], split=config[mode])
    text = [dataset[key] for key in config['key']]
    inputs = tokenizer(*text, truncation=True, padding='max_length', max_length=args.token_length, return_tensors='pt')
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    # concate input ids with plain text tokens
    plain_tokens = torch.randint(0, args.rec_vocab_size, (input_ids.shape[0], args.n_plain_tok))
    input_ids = torch.cat((plain_tokens, input_ids), dim=1)
    input_ids = input_ids[:, :-args.n_plain_tok]
    # set up mask for plain text tokens
    plain_tokens_mask = torch.ones(input_ids.shape[0], args.n_plain_tok)
    attention_mask = torch.cat((plain_tokens_mask, attention_mask), dim=1)
    attention_mask = attention_mask[:, :-args.n_plain_tok]
    inputs = {'input_ids': input_ids.to(args.device), 'attention_mask': attention_mask.to(args.device)}
    labels = dataset['label']
    labels = torch.tensor(labels).to(args.device)
    return inputs, labels

def get_dataset_sep(d_name, tokenizer, args, mode="all"):
    if d_name not in task_dataset_configs:
        raise ValueError(f"Unknown dataset name: {d_name}")
    
    config = task_dataset_configs[d_name]
    # get train dataset
    train_inputs, train_labels, test_inputs, test_labels = None, None, None, None
    if mode == "all" or mode == "train":
        train_inputs, train_labels = load_dataset_sep(config, tokenizer, args, "train")
    # get test dataset
    if mode == "all" or mode == "test":
        test_inputs, test_labels = load_dataset_sep(config, tokenizer, args, "test")
    return train_inputs, train_labels, test_inputs, test_labels

def get_dataset_options(d_name, tokenizer, size, args):
    if d_name not in dataset_configs:
        raise ValueError(f"Unknown dataset name: {d_name}")

    config = dataset_configs[d_name]
    dataset = load_dataset(config['dataset'], config['subset'], split='train') if config['subset'] else load_dataset(config['dataset'], split='train')
    # print(len(dataset))
    subset_indices = np.random.choice(len(dataset), min(size, len(dataset)), replace=False)
    sample_dataset = dataset.select(subset_indices)
    text = sample_dataset[config['key']]
    if d_name == 'daily_dialog':
        temp = []
        for t in text:
            t = " ".join(t)
            temp.append(t)
        text = temp
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=args.token_length, return_tensors='pt')
    outputs = {}
    for name, tensor in inputs.items():
        outputs[name] = tensor.to(args.device)
    return outputs

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from util.parameters import get_args
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # input1 = get_dataset_options('squad', tokenizer, args.denoise_size, args)
    # input2 = get_dataset_options('glue_qqp', tokenizer, args.denoise_size, args)
    # input2 =get_dataset_sep('rajpurkar/squad', tokenizer, args, mode="all")
    input3 = get_dataset_options("tweet_offensive", tokenizer, 10000, args)
    a = 1