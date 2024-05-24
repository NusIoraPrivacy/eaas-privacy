from models.model import *
from data.load_data import *
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from data.dataset import HDF5Dataset, SyntheticDataset, create_mixed_dataset, UniversalDataset
from util.utils import get_embeddings, get_denoise_path
import numpy as np
import time
import os
import h5py
from util.globals import *

def preprocess_data(base_model, args, tokenizer=None):
    if args.denoise_data == "synthetic":
        inputs = synthesize_data(args.denoise_size, args.token_length, tokenizer, args)
        train_dataset = SyntheticDataset(inputs)
    elif args.denoise_data == "mix":
        train_dataset = create_mixed_dataset(args.mixed_data_config_list, tokenizer, args.token_length)
        print("mixed_training_dataset_length: ", len(train_dataset))
    else:
        train_dataset = UniversalDataset(args.denoise_data, 'train', tokenizer, args.denoise_size, args)
        print("train_dataset_length: ", len(train_dataset))
    print("finish loading all data")
    base_model = base_model.to(args.device)
    # get initial embedding
    input_ids = inputs['input_ids'].to(args.device)
    attention_masks = inputs['attention_mask'].to(args.device)
    
    # 创建保存目录
    base_mod_name = args.base_model.split('/')[-1]
    if args.noise_mechanism == "Gaussian":
        if args.denoise_data == "mix":
            denoise_data_name = args.mixed_data_config_list.split(",")
            denoise_data_name = [i.strip()[0] for i in denoise_data_name]
            denoise_data_name = ''.join(denoise_data_name)
            save_dir = os.path.join(args.denoise_data_dir, f"{args.denoise_data}_{denoise_data_name}_noise_{base_mod_name}_{args.noise_mechanism}_{args.noise_std}_{args.clip}")
        else:
            save_dir = os.path.join(args.denoise_data_dir, f"{args.denoise_data}_noise_{base_mod_name}_{args.noise_mechanism}_{args.noise_std}_{args.clip}")
    elif args.noise_mechanism == "ChiDP":
        if args.denoise_data == "mix":
            denoise_data_name = args.mixed_data_config_list.split(",")
            denoise_data_name = [i.strip()[0] for i in denoise_data_name]
            denoise_data_name = ''.join(denoise_data_name)
            save_dir = os.path.join(args.denoise_data_dir, f"{args.denoise_data}_{denoise_data_name}_noise_{base_mod_name}_{args.noise_mechanism}_{args.train_eta}_{args.clip}")
        else:
            save_dir = os.path.join(args.denoise_data_dir, f"{args.denoise_data}_noise_{base_mod_name}_{args.noise_mechanism}_{args.train_eta}_{args.clip}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if "t5" in args.base_model:
        decoder_start_token = tokenizer("classify: ", return_tensors='pt').input_ids
        decoder_start_token = decoder_start_token.to(args.device)
    else:
        decoder_start_token = None

    # 打开HDF5文件进行写入
    with h5py.File(os.path.join(save_dir, 'train_data.hdf5'), 'w') as hf:
        for i in tqdm(range(0, len(input_ids), args.base_batch_size)):
            this_input_ids = input_ids[i:i+args.base_batch_size]
            this_attention_masks = attention_masks[i:i+args.base_batch_size]

            for j in range(args.noise_per_sample):
                this_noises, this_clean_cls_emb, this_noise_cls_emb, _ = get_embeddings(this_input_ids, this_attention_masks, base_model, args, decoder_start_token, "train")
                if args.mask_init:
                    mask = this_attention_masks.unsqueeze(-1).expand_as(this_noises)
                    this_noises = this_noises.masked_fill(mask == 0, 0)
                
                # 为数据创建数据集，或将数据追加到现有数据集
                for name, data in zip(['input_ids', 'attention_masks', 'noises', 'clean_cls_emb', 'noise_cls_emb'],
                                    [this_input_ids.cpu(), this_attention_masks.cpu(), this_noises.cpu(), this_clean_cls_emb.cpu(), this_noise_cls_emb.cpu()]):
                    if name not in hf:
                        maxshape = (None,) + data.shape[1:]
                        hf.create_dataset(name, data=data, maxshape=maxshape, chunks=True)
                    else:
                        hf[name].resize((hf[name].shape[0] + data.shape[0]), axis=0)
                        hf[name][-data.shape[0]:] = data

    print(len(input_ids))
    return len(input_ids)

def load_train(train_h5_path, base_model, args=None):
    # initialize denoise model
    embed_dim = emb_size_dict[args.base_model]
    denoise_mod = eval(args.denoise_model)(d_model=embed_dim, d_out=embed_dim, args=args)
    denoise_mod = denoise_mod.to(args.device)
    optimizer = torch.optim.Adam(denoise_mod.parameters(), lr=0.0001)

    dataset = HDF5Dataset(train_h5_path)

    dataloader = DataLoader(dataset, batch_size=args.denoise_batch_size, shuffle=True, num_workers=6, drop_last=False, pin_memory=True)
    
    # train the model
    base_model = base_model.to(args.device)
    for epoch in range(args.denoise_epochs):
        for input_ids, attention_masks, noises, clean_cls_emb, noise_cls_emb in tqdm(dataloader):
            # 此处的数据已经在一个批次中
            denoise_mod.train()
            input_ids = input_ids.to(args.device)
            attention_masks = attention_masks.to(args.device)
            noises = noises.to(args.device)
            clean_cls_emb = clean_cls_emb.to(args.device)
            noise_cls_emb = noise_cls_emb.to(args.device)

            this_emb = get_token_embedding(input_ids, base_model, args)
            
            if args.mask_init:
                mask_rshp = attention_masks.unsqueeze(-1).expand_as(this_emb)
                this_emb = this_emb.masked_fill(mask_rshp == 0, 0)
            
            if args.mask_attn:
                y_pred = denoise_mod(this_emb, noises, noise_cls_emb, attention_masks)
            else:
                y_pred = denoise_mod(this_emb, noises, noise_cls_emb)
                if args.loss == 'mse':
                    loss_fn = nn.MSELoss()
                    loss = loss_fn(y_pred, clean_cls_emb)
                elif args.loss == 'cosEmb':
                    target = torch.ones(y_pred.size(0)).to(args.device)
                    loss_fn = torch.nn.CosineEmbeddingLoss()
                    loss = loss_fn(y_pred, clean_cls_emb, target)
            

            optimizer.zero_grad()
            loss.backward(retain_graph=True) # <---- retain the graph after backprop
            optimizer.step()           
        print(f'Finished epoch {epoch}, latest loss {loss.item()}')
        base_mod_name = args.base_model.split('/')[-1]
        if args.denoise_data == "mix":
            denoise_data_name = args.mixed_data_config_list.split(",")
            denoise_data_name = [i.strip()[0] for i in denoise_data_name]
            denoise_data_name = ''.join(denoise_data_name)
            denoise_mode_subdir = os.path.join(args.denoise_model_dir, f"denoise_model_{base_mod_name}_{args.denoise_data}_{denoise_data_name}_{args.denoise_model}_{args.comb}_{str(args.att_pool)}_{str(args.mask_init)}_{str(args.mask_attn)}_{str(args.num_heads)}_{str(args.num_layers)}_{str(args.dim_head)}_{args.clip}_eta{str(args.train_eta)}")
        else:
            denoise_mode_subdir = os.path.join(args.denoise_model_dir, f"denoise_model_{base_mod_name}_{args.denoise_data}_{args.denoise_model}_{args.comb}_{str(args.att_pool)}_{str(args.mask_init)}_{str(args.mask_attn)}_{str(args.num_heads)}_{str(args.num_layers)}_{str(args.dim_head)}_{args.clip}_eta{str(args.train_eta)}")
        if not os.path.exists(denoise_mode_subdir):
            os.makedirs(denoise_mode_subdir, exist_ok=True)
        out_path = os.path.join(denoise_mode_subdir, f"epoch_{epoch}")
        torch.save(denoise_mod.state_dict(), out_path)

def train_denoise(base_model, tokenizer=None, args=None):
    '''
    train a denoise model on server side
    '''
    if args.denoise_data == "synthetic":
        inputs = synthesize_data(args.denoise_size, args.token_length, tokenizer, args)
        train_dataset = SyntheticDataset(inputs)
        print("train_dataset_length: ", len(train_dataset))
    elif args.denoise_data == "mix":
        train_dataset = create_mixed_dataset(args.mixed_data_config_list, tokenizer, args.token_length)
        print("mixed_training_dataset_length: ", len(train_dataset))
    else:
        train_dataset = UniversalDataset(args.denoise_data, 'train', tokenizer, args.denoise_size, args)
        print("train_dataset_length: ", len(train_dataset))
    print("finish loading all data")
    
    base_model = base_model.to(args.device)
    base_model = base_model.eval()
    
    # initialize denoise model
    embed_dim = emb_size_dict[args.base_model]
    denoise_mod = eval(args.denoise_model)(d_model=embed_dim, d_out=embed_dim, args=args)
    if args.use_ft_base:
        model_path = get_denoise_path(args, 1)
        denoise_mod.load_state_dict(torch.load(model_path, map_location=args.device))
    denoise_mod = denoise_mod.to(args.device)
    optimizer = torch.optim.Adam(denoise_mod.parameters(), lr=0.0001)
    # subsample train dataset
    if args.use_ft_base:
        sample_indices = np.random.choice(len(train_dataset), args.ft_denoise_sample_size, replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, sample_indices)
    # 创建 DataLoader 实例
    dataloader = DataLoader(train_dataset, batch_size=args.base_batch_size, shuffle=False, num_workers=0, drop_last=True, pin_memory=True)
    # scaler = GradScaler()
    epoch_pbar = tqdm(range(args.denoise_epochs), desc="Epochs")
    if "t5" in args.base_model:
        decoder_start_token = tokenizer("classify: ", return_tensors='pt').input_ids
        decoder_start_token = decoder_start_token.to(args.device)
    else:
        decoder_start_token = None
    total_t = 0
    t1 = time.time()
    for epoch in epoch_pbar:
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1} Batches"), start=1):
            input_ids = batch['input_ids'].to(args.device)
            attention_masks = batch['attention_mask'].to(args.device)
            for j in range(args.noise_per_sample):
                noises, clean_cls_emb, noise_cls_emb, init_emb = get_embeddings(input_ids, attention_masks, base_model, args, decoder_start_token, "train")
                if args.mask_init:
                    mask = attention_masks.unsqueeze(-1).expand_as(noises)
                    noises = noises.masked_fill(mask == 0, 0)
                    init_emb = init_emb.masked_fill(mask == 0, 0)
                
                noises = noises.to(dtype=args.denoise_precision)
                init_emb = init_emb.to(dtype=args.denoise_precision)
                clean_cls_emb = clean_cls_emb.to(dtype=args.denoise_precision)
                noise_cls_emb = noise_cls_emb.to(dtype=args.denoise_precision)
                # training denoise model
                optimizer.zero_grad()
                if args.mask_attn:
                    y_pred = denoise_mod(init_emb, noises, noise_cls_emb, attention_masks)
                else:
                    y_pred = denoise_mod(init_emb, noises, noise_cls_emb)
                
                if args.loss == 'mse':
                    loss_fn = nn.MSELoss()
                    loss = loss_fn(y_pred, clean_cls_emb)
                elif args.loss == 'cosEmb':
                    target = torch.ones(y_pred.size(0)).to(args.device)
                    loss_fn = torch.nn.CosineEmbeddingLoss()
                    loss = loss_fn(y_pred, clean_cls_emb, target)
                
                loss.backward() 
                optimizer.step()

                if i % 50 == 0:
                    print(f'Finished epoch {epoch} batch {i}, latest loss {loss.item()}')
            
            if i >= args.time_steps and args.cal_time:
                if epoch == args.denoise_epochs-1:
                    t2 = time.time()
                    total_t = (t2 - t1)/args.time_steps*len(dataloader)
                    print(f"Training time for {args.base_model} is {total_t} seconds")
                break

        epoch_pbar.set_description(f"Completed Epoch {epoch+1}")             
        print(f'Finished epoch {epoch}, latest loss {loss.item()}')
        
        base_mod_name = args.base_model.split('/')[-1]
        if args.denoise_data == "mix":
            denoise_data_name = args.mixed_data_config_list.split(",")
            denoise_data_name = [i.strip()[0] for i in denoise_data_name]
            denoise_data_name = ''.join(denoise_data_name)
            denoise_mode_subdir = os.path.join(args.denoise_model_dir, f"denoise_model_{base_mod_name}_{args.denoise_data}_{denoise_data_name}_{args.denoise_model}_{args.comb}_{str(args.att_pool)}_{str(args.mask_init)}_{str(args.mask_attn)}_{str(args.num_heads)}_{str(args.num_layers)}_{str(args.dim_head)}_{args.clip}_eta{str(args.train_eta)}")
        else:
            denoise_mode_subdir = os.path.join(args.denoise_model_dir, f"denoise_model_{base_mod_name}_{args.denoise_data}_{args.denoise_model}_{args.comb}_{str(args.att_pool)}_{str(args.mask_init)}_{str(args.mask_attn)}_{str(args.num_heads)}_{str(args.num_layers)}_{str(args.dim_head)}_{args.clip}_eta{str(args.train_eta)}")
        if args.use_ft_base:
            denoise_mode_subdir = f"{denoise_mode_subdir}_ft_{args.ft_denoise_sample_size}"
        if not os.path.exists(denoise_mode_subdir):
            os.makedirs(denoise_mode_subdir, exist_ok=True)
        out_path = os.path.join(denoise_mode_subdir, f"epoch_{epoch}")
        torch.save(denoise_mod.state_dict(), out_path)
        
    epoch_pbar.close()

def test_denoise(clean_cls_emb, noise_cls_emb, denoise_cls_emb, args):
    # inputs = inputs.to(args.device)
    # inputs = {name: tensor.to(args.device) for name, tensor in inputs.items()}
    cosine_similarity = torch.nn.functional.cosine_similarity
    metric = nn.MSELoss()

    # compute mse before denoise
    this_noise_mse = metric(clean_cls_emb, noise_cls_emb)
    # compute cosine similarity before denoise
    this_noise_sim = cosine_similarity(clean_cls_emb, noise_cls_emb).mean()
    
    # compute mse after denoise
    this_denoise_mse = metric(clean_cls_emb, denoise_cls_emb)
    # compute cosine similarity after denoise
    this_denoise_sim = cosine_similarity(clean_cls_emb, denoise_cls_emb).mean()
    
    return this_noise_mse, this_denoise_mse, this_noise_sim, this_denoise_sim