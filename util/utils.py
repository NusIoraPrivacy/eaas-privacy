# import sys
# import os
# current = os.path.dirname(os.path.realpath(__file__))
# parent = os.path.dirname(current)
# sys.path.append(parent)
import torch
from tqdm import tqdm
from data.load_data import sample_noise_Gaussian, sample_noise_Chi
import os
import h5py
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, 
                          AutoModel, GPT2Tokenizer, GPT2Model, 
                          OPTForSequenceClassification,
                          AutoModelForCausalLM, BertModel,
                          T5Model,)
from util.globals import *

def get_embeddings(input_ids, attention_mask, model, args, decoder_start_token=None, mode="train"):
    '''
    server compute the embeddings given to the client
    '''
    #  get initial embeddings
    init_emb = get_token_embedding(input_ids, model, args)
    # sample noise
    if args.noise_mechanism == "Gaussian":
        noise_std = args.train_noise_std if mode == "train" else args.test_noise_std
        noises = sample_noise_Gaussian(init_emb.shape, noise_std, args.device)
    elif args.noise_mechanism == "ChiDP":
        eta = args.train_eta if mode == "train" else args.test_eta
        noises = sample_noise_Chi(init_emb.shape, eta, args.device)
    noise_init_emb = init_emb + noises
    if args.clip == "element":
        lower, upper = emb_range_dict[args.base_model]
        noise_init_emb = torch.clamp(noise_init_emb, lower, upper)
        noises = noise_init_emb - init_emb
    elif args.clip == "norm":
        max_norm = emb_norm_dict[args.base_model]
        all_norms = torch.norm(noise_init_emb, p=2, dim=-1)
        noise_init_emb = noise_init_emb * torch.clamp(max_norm / all_norms, max=1).unsqueeze(-1)
        noises = noise_init_emb - init_emb
    # obtain final embeddings without noise
    with torch.no_grad():
        if "t5" in args.base_model:
            decoder_start_token = decoder_start_token.repeat(init_emb.shape[0], 1)
            outputs = model(inputs_embeds=init_emb, attention_mask=attention_mask, decoder_input_ids=decoder_start_token, output_hidden_states=True)
        else:
            outputs = model(inputs_embeds=init_emb, attention_mask=attention_mask, output_hidden_states=True)
        clean_cls_embs = get_cls_embedding(outputs, attention_mask, args)

    # get final embeddings with noise
    with torch.no_grad():
        if "t5" in args.base_model:
            noise_outputs = model(inputs_embeds=noise_init_emb, attention_mask=attention_mask, decoder_input_ids=decoder_start_token, output_hidden_states=True)
        else:
            noise_outputs = model(inputs_embeds=noise_init_emb, attention_mask=attention_mask, output_hidden_states=True)
        noise_cls_embs = get_cls_embedding(noise_outputs, attention_mask, args)
        
    return noises, clean_cls_embs, noise_cls_embs, init_emb

def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    
def str2type(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, torch.dtype):
        return v
    if "float32" in v.lower():
        return torch.float32
    elif "float16" in v.lower():
        return torch.float16

def pt_to_hdf5(args):
    base_mod_name = args.base_model.split('/')[-1]
    if args.noise_mechanism == "Gaussian":
        denoise_data_dir = os.path.join(args.denoise_data_dir, f"{args.denoise_data}_noise_{base_mod_name}_{args.noise_mechanism}_{args.noise_std}")
    elif args.noise_mechanism == "ChiDP":
        denoise_data_dir = os.path.join(args.denoise_data_dir, f"{args.denoise_data}_noise_{base_mod_name}_{args.noise_mechanism}_{args.eta}")
    input_ids = os.path.join(denoise_data_dir, "input_ids.pt")
    attn_masks = os.path.join(denoise_data_dir, "attention_masks.pt")
    noises = os.path.join(denoise_data_dir, "noises.pt")
    clean_cls_emb = os.path.join(denoise_data_dir, "clean_cls_emb.pt")
    noise_cls_emb = os.path.join(denoise_data_dir, "noise_cls_emb.pt")
    input_ids = torch.load(input_ids)
    attention_masks = torch.load(attn_masks)
    noises = torch.load(noises)
    clean_cls_emb = torch.load(clean_cls_emb)
    noise_cls_emb = torch.load(noise_cls_emb)

    print("start converting pt to hdf5 files")
    with h5py.File(os.path.join(denoise_data_dir, 'train_data.hdf5'), 'w') as hf:
         # 为数据创建数据集，或将数据追加到现有数据集
        for name, data in zip(['input_ids', 'attention_masks', 'noises', 'clean_cls_emb', 'noise_cls_emb'],
                            [input_ids, attention_masks, noises, clean_cls_emb, noise_cls_emb]):
            maxshape = (None,) + data.shape[1:]
            hf.create_dataset(name, data=data, maxshape=maxshape, chunks=True)
    print("finish converting pt to hdf5 files")

def get_cls_embedding(outputs, attention_mask, args):
    if args.base_model =="stevhliu/my_awesome_model":
        hid_states = outputs.hidden_states
        cls_embs = hid_states[-1][:,0,:]
    elif args.base_model in ("bert-base-uncased", "bert-large-uncased"):
        hid_states = outputs.hidden_states
        cls_embs = hid_states[-1][:,0,:]
    elif "gpt2" in args.base_model:
        hid_states = outputs.last_hidden_state
        sum_attention_mask = attention_mask.sum(dim=1)
        # 防止序列长度为0导致的-1索引
        last_pad = torch.where(sum_attention_mask > 0, sum_attention_mask - 1, torch.zeros_like(sum_attention_mask))
        last_pad = last_pad.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hid_states.size(2))
        cls_embs = torch.gather(hid_states, index=last_pad, dim=1)
        cls_embs = cls_embs.squeeze()
    elif "t5" in args.base_model:
        hid_states = outputs.last_hidden_state
        cls_embs = hid_states[:,0,:]
    elif any(model in args.base_model for model in ['opt', 'llama']):
        hid_states = outputs.hidden_states[-1]
        last_pad = attention_mask.sum(dim=1) - 1
        last_pad = last_pad.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, hid_states.shape[-1])
        cls_embs = torch.gather(hid_states, index=last_pad, dim=1)
        cls_embs = cls_embs.squeeze()
    return cls_embs

def get_token_embedding(token_id, model, args, squeeze=False):
    """get the token embedding given the input ids"""
    with torch.no_grad():
        if args.base_model =="stevhliu/my_awesome_model":
            embeddings = model.distilbert.embeddings.word_embeddings(token_id)
            # embeddings = model.distilbert.embeddings(token_id)
        elif args.base_model in ("bert-base-uncased", "bert-large-uncased"):
            embeddings = model.embeddings.word_embeddings(token_id)
        elif args.base_model in ("THUDM/chatglm2-6b-int4", "THUDM/chatglm2-6b"):
            transf = model.transformer
            embeddings = transf.embedding.word_embeddings(token_id)
        elif 'gpt2' in args.base_model:
            embeddings = model.wte(token_id)
        elif 'opt' in args.base_model:
            try:
                embeddings = model.model.decoder.embed_tokens(token_id)
            except:
                embeddings = model.decoder.embed_tokens(token_id)
        elif 'llama' in args.base_model:
            try:
                embeddings = model.model.embed_tokens(token_id)
            except:
                embeddings = model.embed_tokens(token_id)
        elif 't5' in args.base_model:
            embeddings = model.encoder.embed_tokens(token_id)
        if squeeze:
            embeddings = embeddings.squeeze()
    return embeddings

def get_closest_token(embedding, tokenizer, model, args):
    """Find the word with the closest embedding."""
    closest_token = None
    if 'gpt2' in args.base_model:
        vocabulary = tokenizer.get_vocab()
    else:
        vocabulary = tokenizer.vocab
    token_ids = [token_id for _, token_id in vocabulary.items()]
    token_ids = torch.tensor(token_ids).to(args.device)
    word_embeddings = get_token_embedding(token_ids, model, args, squeeze=True)
    embedding = embedding.unsqueeze(dim=0)
    embedding = embedding.expand(word_embeddings.size())
    distance = torch.norm(embedding - word_embeddings, 2, dim=1)
    closest_idx = distance.argmin()
    closest_token = token_ids[closest_idx]
    return closest_token.item()

def text2text_priv(inputs, tokenizer, model, args):
    input_ids = inputs['input_ids']
    attn_masks = inputs['attention_mask']
    init_embeddings = get_token_embedding(input_ids, model, args)
    # sample noise
    if args.noise_mechanism == "Gaussian":
        noises = sample_noise_Gaussian(init_embeddings.shape, args.test_noise_std, args.device)
    elif args.noise_mechanism == "ChiDP":
        noises = sample_noise_Chi(init_embeddings.shape, args.test_eta, args.device)
    init_embeddings = init_embeddings + noises
    for i in tqdm(range(len(input_ids))):
        this_mask = attn_masks[i]
        this_embeds = init_embeddings[i]
        this_embeds = this_embeds[this_mask == 1]
        for j in range(len(this_embeds)):
            embed = this_embeds[j]
            closest_token = get_closest_token(embed, tokenizer, model, args)
            input_ids[i, j] = closest_token
    return {"input_ids": input_ids, 'attention_mask': attn_masks}

def get_pretrained_model(args):
    if args.base_model =="stevhliu/my_awesome_model":
        base_model = AutoModelForSequenceClassification.from_pretrained(args.base_model)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    elif args.base_model in ("bert-base-uncased", "bert-large-uncased"):
        base_model = BertModel.from_pretrained(args.base_model)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    elif args.base_model in ("THUDM/chatglm2-6b-int4", "THUDM/chatglm2-6b"):
        base_model = AutoModel.from_pretrained(args.base_model, trust_remote_code=True).half() # FP16 by default
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    elif 'gpt2' in args.base_model:
        tokenizer = GPT2Tokenizer.from_pretrained(args.base_model)
        tokenizer.pad_token = tokenizer.eos_token
        base_model = GPT2Model.from_pretrained(args.base_model)
    elif 'opt' in args.base_model:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        base_model = OPTForSequenceClassification.from_pretrained(args.base_model)
    elif 'llama' in args.base_model:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        tokenizer.pad_token = tokenizer.eos_token
        if args.llama_dir is not None:
            base_model = AutoModelForCausalLM.from_pretrained(args.llama_dir, torch_dtype=args.base_precision)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=args.base_precision)
    elif 't5' in args.base_model:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        base_model = T5Model.from_pretrained(args.base_model)
    return tokenizer, base_model

def get_ft_pretrained_model(args):
    if args.base_model in ("bert-base-uncased", "bert-large-uncased"):
        base_model = AutoModelForSequenceClassification.from_pretrained(args.base_model)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    return tokenizer, base_model

def get_finetuned_model(args):
    mod_path = f"{args.denoise_model_dir}/finetune/{args.base_model}"
    # if not os.path.exists(mod_path):
    #     os.makedirs(mod_path, exist_ok=True)
    if args.base_model in ("bert-base-uncased", "bert-large-uncased"):
        base_model = BertModel.from_pretrained(mod_path)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    return tokenizer, base_model

def get_denoise_path(args, d_epoch, use_ft=False):
    if args.ckpt_path is None:
        base_mod_name = args.base_model.split('/')[-1]
        if args.denoise_data == "mix":
            denoise_data_name = args.mixed_data_config_list.split(",")
            denoise_data_name = [i.strip()[0] for i in denoise_data_name]
            denoise_data_name = ''.join(denoise_data_name)
            denoise_mode_subdir = os.path.join(args.denoise_model_dir, f"denoise_model_{base_mod_name}_{args.denoise_data}_{denoise_data_name}_{args.denoise_model}_{args.comb}_{str(args.att_pool)}_{str(args.mask_init)}_{str(args.mask_attn)}_{str(args.num_heads)}_{str(args.num_layers)}_{str(args.dim_head)}_{args.clip}_eta{str(args.train_eta)}")
        else:
            denoise_mode_subdir = os.path.join(args.denoise_model_dir, f"denoise_model_{base_mod_name}_{args.denoise_data}_{args.denoise_model}_{args.comb}_{str(args.att_pool)}_{str(args.mask_init)}_{str(args.mask_attn)}_{str(args.num_heads)}_{str(args.num_layers)}_{str(args.dim_head)}_{args.clip}_eta{str(args.train_eta)}")
        if use_ft and args.ft_denoise_sample_size>0:
            denoise_mode_subdir = f"{denoise_mode_subdir}_ft_{args.ft_denoise_sample_size}"
        model_path = os.path.join(denoise_mode_subdir, f"epoch_{d_epoch}")
    else:
        model_path = args.ckpt_path
    return model_path

def get_token_distribution(dataloader, tokenizer, vocabulary, args):
    # dataloader: downstream dataloader
    # obtain max id
    token_ids = [token_id for _, token_id in vocabulary.items()]
    max_id = max(token_ids)
    # get frequency dictionary
    freq_dict = {}
    for i, batch in enumerate(tqdm(dataloader), start=1):
        input_ids = torch.cat(batch['input_ids'], 1)
        attention_masks = torch.cat(batch['attention_mask'], 1)
        labels = batch['labels']
        input_ids = input_ids.masked_fill(attention_masks == 0, 0)
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            this_input_ids = input_ids[labels==label]
            this_freq = torch.bincount(torch.flatten(this_input_ids), minlength=max_id)
            if label.item() in freq_dict.keys():
                freq_dict[label.item()] += this_freq
            else:
                freq_dict[label.item()] = this_freq
    return freq_dict

def get_ui_top(dataloader, tokenizer, args):
    if 'gpt2' in args.base_model:
        vocabulary = tokenizer.get_vocab()
    else:
        vocabulary = tokenizer.vocab
    top_k = int(len(vocabulary) * 0.01)
    # get frequency dictionary
    freq_dict = get_token_distribution(dataloader, tokenizer, vocabulary, args)
    esp = 1e10-5 # avoid dividing by zero
    # compute ui values for each toekn
    ui_dict = {}
    for label1 in freq_dict:
        this_freq = freq_dict[label1]
        this_ui = 0
        for label2 in freq_dict:
            if label2 != label1:
                other_freq = freq_dict[label2] + esp
                this_ui += torch.log(this_freq/other_freq)
        ui_dict[label1] = this_ui
    # obtain the tokens with top k ui
    top_token_dict = {}
    for label in ui_dict:
        output = torch.topk(ui_dict[label], top_k)
        top_tokens = output.indices
        top_tokens = top_tokens[top_tokens!=0]
        top_token_dict[label] = top_tokens
    return top_token_dict

def text2text_priv_ui(inputs, labels, tokenizer, model, top_ui_dict, args):
    input_ids = inputs['input_ids']
    attn_masks = inputs['attention_mask']
    init_embeddings = get_token_embedding(input_ids, model, args)
    # sample noise
    if args.noise_mechanism == "Gaussian":
        noises = sample_noise_Gaussian(init_embeddings.shape, args.test_noise_std, args.device)
    elif args.noise_mechanism == "ChiDP":
        noises = sample_noise_Chi(init_embeddings.shape, args.test_eta, args.device)
    init_embeddings = init_embeddings + noises
    for i in tqdm(range(len(input_ids))):
        this_mask = attn_masks[i]
        this_embeds = init_embeddings[i]
        this_embeds = this_embeds[this_mask == 1]
        this_label = labels[i]
        for j in range(len(this_embeds)):
            if input_ids[i, j] not in top_ui_dict[this_label.item()]:
                embed = this_embeds[j]
                closest_token = get_closest_token(embed, tokenizer, model, args)
                input_ids[i, j] = closest_token
    return {"input_ids": input_ids, 'attention_mask': attn_masks}