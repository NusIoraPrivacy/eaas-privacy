import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model import serverDenoiseModel
from data.dataset import SyntheticDataset, create_mixed_dataset, UniversalDataset, SyntheticDataset
from data.load_data import synthesize_data, sample_noise_Gaussian, sample_noise_Chi

from util.globals import *
from util.utils import get_token_embedding, get_pretrained_model
from util.parameters import get_args

import os

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
    # obtain token embedding matrix
    if 'gpt2' in args.base_model:
        vocabulary = tokenizer.get_vocab()
    else:
        vocabulary = tokenizer.vocab
    token_ids = [token_id for _, token_id in vocabulary.items()]
    token_ids = torch.tensor(token_ids).to(args.device)
    word_embeddings = get_token_embedding(token_ids, base_model, args, squeeze=True)
    # initialize denoise model
    embed_dim = emb_size_dict[args.base_model]
    denoise_mod = serverDenoiseModel(d_model=embed_dim, token_embedding=word_embeddings, args=args)
    denoise_mod = denoise_mod.to(args.device)
    optimizer = torch.optim.Adam(denoise_mod.parameters(), lr=0.0001)
    # 创建 DataLoader 实例
    dataloader = DataLoader(train_dataset, batch_size=args.base_batch_size, shuffle=False, num_workers=0, drop_last=True, pin_memory=True)
    # scaler = GradScaler()
    epoch_pbar = tqdm(range(args.denoise_epochs), desc="Epochs")
    if "t5" in args.base_model:
        decoder_start_token = tokenizer("classify: ", return_tensors='pt').input_ids
        decoder_start_token = decoder_start_token.to(args.device)
    else:
        decoder_start_token = None
    # get word embeddings
    
    for epoch in epoch_pbar:
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1} Batches"), start=1):
            input_ids = batch['input_ids'].to(args.device)
            attention_masks = batch['attention_mask'].to(args.device)
            for j in range(args.noise_per_sample):
                init_emb = get_token_embedding(input_ids, base_model, args)
                # sample noise
                if args.noise_mechanism == "Gaussian":
                    noise_std = args.train_noise_std
                    noises = sample_noise_Gaussian(init_emb.shape, noise_std, args.device)
                elif args.noise_mechanism == "ChiDP":
                    eta = args.train_eta
                    noises = sample_noise_Chi(init_emb.shape, eta, args.device)
                noise_init_emb = init_emb + noises
                mask = attention_masks.unsqueeze(-1).expand_as(noises)
                noises = noises.masked_fill(mask == 0, 0)
                init_emb = init_emb.masked_fill(mask == 0, 0)
                
                # training denoise model
                optimizer.zero_grad()
                _, loss = denoise_mod(noise_init_emb, attention_masks, input_ids)
                
                loss.backward() 
                optimizer.step()

                if i % 50 == 0:
                    print(f'Finished epoch {epoch} batch {i}, latest loss {loss.item()}')

        epoch_pbar.set_description(f"Completed Epoch {epoch+1}")             
        print(f'Finished epoch {epoch}, latest loss {loss.item()}')
        
        base_mod_name = args.base_model.split('/')[-1]
        if args.denoise_data == "mix":
            denoise_data_name = args.mixed_data_config_list.split(",")
            denoise_data_name = [i.strip()[0] for i in denoise_data_name]
            denoise_data_name = ''.join(denoise_data_name)
            denoise_mode_subdir = os.path.join(args.denoise_model_dir, f"server_denoise_model_{base_mod_name}_{args.denoise_data}_{denoise_data_name}_{args.denoise_model}_{args.comb}_{str(args.att_pool)}_{str(args.mask_init)}_{str(args.mask_attn)}_{str(args.num_heads)}_{str(args.num_layers)}_{str(args.dim_head)}_{args.clip}_eta{str(args.train_eta)}")
        else:
            denoise_mode_subdir = os.path.join(args.denoise_model_dir, f"server_denoise_model_{base_mod_name}_{args.denoise_data}_{args.denoise_model}_{args.comb}_{str(args.att_pool)}_{str(args.mask_init)}_{str(args.mask_attn)}_{str(args.num_heads)}_{str(args.num_layers)}_{str(args.dim_head)}_{args.clip}_eta{str(args.train_eta)}")
        if not os.path.exists(denoise_mode_subdir):
            os.makedirs(denoise_mode_subdir, exist_ok=True)
        out_path = os.path.join(denoise_mode_subdir, f"epoch_{epoch}")
        torch.save(denoise_mod.state_dict(), out_path)
        
    epoch_pbar.close()
    return denoise_mod

def accuracy(prediction, true, masks):
    mask_num = torch.sum(masks == 0)
    match_num = torch.sum(prediction==true)
    match_num = match_num - mask_num
    total_num = prediction.shape[0] * prediction.shape[1]
    total_num = total_num - mask_num
    acc = match_num/total_num
    return acc, match_num, total_num

def random_guessing(args):
    SAMPLE_SIZE = 10000
    tokenizer, base_model = get_pretrained_model(args)
    base_model = base_model.to(args.device)
    ### load data
    # load task data
    test_dataset = UniversalDataset(args.task, 'train', tokenizer, SAMPLE_SIZE, args)
    dataloader = DataLoader(test_dataset, batch_size=args.base_batch_size, shuffle=False, num_workers=0, drop_last=True, pin_memory=True)
    print("length of testing data: ", len(test_dataset))
    # obtain token embedding matrix
    if 'gpt2' in args.base_model:
        vocabulary = tokenizer.get_vocab()
    else:
        vocabulary = tokenizer.vocab
    # Load the trained denoise model
    match_num = 0 
    total_num = 0
    with tqdm(
            total=len(dataloader), unit='batch'
        ) as pbar:
        for i, batch in enumerate(dataloader, start=1):
            input_ids = batch['input_ids']
            attention_masks = batch['attention_mask']
            # random guessing
            pred_tokens = torch.randint(0, len(vocabulary), size=input_ids.shape)
            pred_tokens = pred_tokens.masked_fill(attention_masks == 0, 0)
            input_ids = input_ids.masked_fill(attention_masks == 0, 0)
            # compute accuracy
            _, this_match_num, this_total_num = accuracy(pred_tokens, input_ids, attention_masks)
            match_num += this_match_num
            total_num += this_total_num
            pbar.update(1)
            acc = match_num/total_num
            pbar.set_postfix(acc=acc)
        print(f"accuracy for is {acc}")

def test_denoise(args, denoise_model=None):
    SAMPLE_SIZE = 1000
    tokenizer, base_model = get_pretrained_model(args)
    base_model = base_model.to(args.device)
    ### load data
    # load task data
    test_dataset = UniversalDataset(args.task, 'train', tokenizer, SAMPLE_SIZE, args)
    dataloader = DataLoader(test_dataset, batch_size=args.base_batch_size, shuffle=False, num_workers=0, drop_last=True, pin_memory=True)
    print("length of testing data: ", len(test_dataset))
    # obtain token embedding matrix
    if 'gpt2' in args.base_model:
        vocabulary = tokenizer.get_vocab()
    else:
        vocabulary = tokenizer.vocab
    token_ids = [token_id for _, token_id in vocabulary.items()]
    token_ids = torch.tensor(token_ids).to(args.device)
    word_embeddings = get_token_embedding(token_ids, base_model, args, squeeze=True)
    # initialize denoise model
    embed_dim = emb_size_dict[args.base_model]
    # Load the trained denoise model
    for d_epoch in range(args.denoise_epochs):
        print(f"Test for denoise model of epoch {d_epoch}")
        # load denoise model
        if denoise_model == None:
            denoise_model = serverDenoiseModel(d_model=embed_dim, token_embedding=word_embeddings, args=args)  # Set the required parameters
            base_mod_name = args.base_model.split('/')[-1]
            if args.denoise_data == "mix":
                denoise_data_name = args.mixed_data_config_list.split(",")
                denoise_data_name = [i.strip()[0] for i in denoise_data_name]
                denoise_data_name = ''.join(denoise_data_name)
                denoise_mode_subdir = os.path.join(args.denoise_model_dir, f"server_denoise_model_{base_mod_name}_{args.denoise_data}_{denoise_data_name}_{args.denoise_model}_{args.comb}_{str(args.att_pool)}_{str(args.mask_init)}_{str(args.mask_attn)}_{str(args.num_heads)}_{str(args.num_layers)}_{str(args.dim_head)}_{args.clip}_eta{str(args.train_eta)}")
            else:
                denoise_mode_subdir = os.path.join(args.denoise_model_dir, f"server_denoise_model_{base_mod_name}_{args.denoise_data}_{args.denoise_model}_{args.comb}_{str(args.att_pool)}_{str(args.mask_init)}_{str(args.mask_attn)}_{str(args.num_heads)}_{str(args.num_layers)}_{str(args.dim_head)}_{args.clip}_eta{str(args.train_eta)}")
            if not os.path.exists(denoise_mode_subdir):
                os.makedirs(denoise_mode_subdir, exist_ok=True)
            denoise_path = os.path.join(denoise_mode_subdir, f"epoch_{d_epoch}")
            denoise_model.load_state_dict(torch.load(denoise_path, map_location=args.device))
        denoise_model.to(args.device)
        match_num = 0 
        total_num = 0
        with tqdm(
                total=len(dataloader), desc=f'Epoch {d_epoch + 1}/{args.denoise_epochs}', unit='batch'
            ) as pbar:
            for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {d_epoch+1} Batches"), start=1):
                input_ids = batch['input_ids'].to(args.device)
                attention_masks = batch['attention_mask'].to(args.device)
                init_emb = get_token_embedding(input_ids, base_model, args)
                # sample noise
                if args.noise_mechanism == "Gaussian":
                    noise_std = args.train_noise_std
                    noises = sample_noise_Gaussian(init_emb.shape, noise_std, args.device)
                elif args.noise_mechanism == "ChiDP":
                    eta = args.train_eta
                    noises = sample_noise_Chi(init_emb.shape, eta, args.device)
                noise_init_emb = init_emb + noises
                mask = attention_masks.unsqueeze(-1).expand_as(noises)
                noises = noises.masked_fill(mask == 0, 0)
                init_emb = init_emb.masked_fill(mask == 0, 0)
                
                # test denoise model
                logits = denoise_model(noise_init_emb, attention_masks) # [batch size, vob size, token length]
                pred_tokens = torch.argmax(logits, dim=1)
                pred_tokens = pred_tokens.masked_fill(attention_masks == 0, 0)
                input_ids = input_ids.masked_fill(attention_masks == 0, 0)
                # compute accuracy
                _, this_match_num, this_total_num = accuracy(pred_tokens, input_ids, attention_masks)
                match_num += this_match_num
                total_num += this_total_num
                pbar.update(1)
                acc = match_num/total_num
                pbar.set_postfix(acc=acc)
        print(f"accuracy for epoch {d_epoch} is {acc}")

if __name__ == "__main__":
    args = get_args()
    # tokenizer, base_model = get_pretrained_model(args)
    # denoise_mod = train_denoise(base_model= base_model, args=args, tokenizer=tokenizer)
    # test_denoise(args, denoise_mod)
    # models = ["bert-base-uncased", "bert-large-uncased", "stevhliu/my_awesome_model"]
    models = ["stevhliu/my_awesome_model"]
    for model in models:
        print(f"Testing for model {model}")
        args.task = "squad"
        args.base_model = model
        random_guessing(args)
        print(f"Finish testing for model {model}")