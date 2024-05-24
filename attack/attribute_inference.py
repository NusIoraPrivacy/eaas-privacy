from datasets import DatasetDict
from data.load_data import get_dataset, sample_noise_Gaussian, sample_noise_Chi
from models.model import *
from util.globals import *
from util.parameters import get_args
from models.model import *
from util.utils import text2text_priv,get_pretrained_model,get_cls_embedding, get_token_embedding
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
import torch
import pandas as pd
import os

def get_noisy_embedding(inputs, model, args):
    input_ids = inputs['input_ids']
    attn_masks = inputs['attention_mask']
    attn_masks = attn_masks.to("cpu")
    init_embeddings = []
    for i in range(0, len(input_ids), args.base_batch_size):
        this_input_ids = input_ids[i:i+args.base_batch_size]
        this_init_embeddings = get_token_embedding(this_input_ids, model, args)
        this_init_embeddings = this_init_embeddings.to("cpu")
        init_embeddings.append(this_init_embeddings)
    init_embeddings = torch.cat(init_embeddings)
    print(init_embeddings.shape)
    # sample noise
    if args.noise_mechanism == "Gaussian":
        noises = sample_noise_Gaussian(init_embeddings.shape, args.test_noise_std, "cpu")
    elif args.noise_mechanism == "ChiDP":
        if args.test_eta > 0:
            noises = sample_noise_Chi(init_embeddings.shape, args.test_eta, "cpu")
        else:
            noises = 0
    init_embeddings = init_embeddings + noises
    return init_embeddings, attn_masks

# def get_embedding(inputs, model, decoder_start_token=None):
#     '''
#     Compute the embeddings for batches of input data.
#     '''
#     cls_embs = []
#     with torch.no_grad():
#         for i in range(0, len(inputs['input_ids']), args.base_batch_size):
#             input_ids_batch = inputs['input_ids'][i:i+args.base_batch_size]
#             att_masks_batch = inputs['attention_mask'][i:i+args.base_batch_size]
            
#             if "t5" in args.base_model and decoder_start_token is not None:
#                 decoder_start_token_batch = decoder_start_token.repeat(input_ids_batch.shape[0], 1)
#                 outputs = model(input_ids=input_ids_batch, 
#                                 attention_mask=att_masks_batch, 
#                                 output_hidden_states=True, 
#                                 decoder_input_ids=decoder_start_token_batch)
#             else:
#                 outputs = model(input_ids=input_ids_batch, 
#                                 attention_mask=att_masks_batch, 
#                                 output_hidden_states=True)
            
#             batch_cls_embs = get_cls_embedding(outputs, att_masks_batch, args)
#             if "cuda" in args.device:
#                 batch_cls_embs = batch_cls_embs.cpu()
#             cls_embs.append(batch_cls_embs)

#     cls_embs = torch.cat(cls_embs, 0)
#     return cls_embs


def get_embedding(token_embeddings, attn_masks, model, args, decoder_start_token=None):
    '''
    Compute the embeddings for batches of input data.
    '''
    cls_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(token_embeddings), args.base_batch_size)):
            token_embeds_batch = token_embeddings[i:i+args.base_batch_size].to(args.device)
            att_masks_batch = attn_masks[i:i+args.base_batch_size].to(args.device)
            
            if "t5" in args.base_model and decoder_start_token is not None:
                decoder_start_token_batch = decoder_start_token.repeat(token_embeds_batch.shape[0], 1)
                outputs = model(inputs_embeds=token_embeds_batch, 
                                attention_mask=att_masks_batch, 
                                output_hidden_states=True, 
                                decoder_input_ids=decoder_start_token_batch)
            else:
                outputs = model(inputs_embeds=token_embeds_batch, 
                                attention_mask=att_masks_batch, 
                                output_hidden_states=True)
            
            batch_cls_embs = get_cls_embedding(outputs, att_masks_batch, args)
            if "cuda" in args.device:
                batch_cls_embs = batch_cls_embs.cpu()
            cls_embs.append(batch_cls_embs)

    cls_embs = torch.cat(cls_embs, 0)
    return cls_embs


def attribute_inference_attack(train_cls_embs, test_cls_embs, train_labels, test_labels):
    # initialize the attack model
    input_dim = train_cls_embs.shape[-1]
    n_classes = len(torch.unique(train_labels))
    mlp_model = AttributeInferenceMLP(input_dim, n_classes).to(args.device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)

    epoch_accuracies = []
    epoch_losses = []
    f1_scores = []

    for epoch in range(args.cls_epochs):
        mlp_model.train()
        for i in tqdm(range(0, len(train_cls_embs), args.cls_batch_size)):
            Xbatch = train_cls_embs[i:i+args.cls_batch_size].to(args.device)
            ybatch = train_labels[i:i+args.cls_batch_size].to(args.device)

            optimizer.zero_grad()
            y_pred = mlp_model(Xbatch)
            loss = loss_fn(y_pred, ybatch)
            loss.backward()
            optimizer.step()

        # Evaluation
        mlp_model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for i in range(0, len(test_cls_embs), args.cls_batch_size):
                Xbatch = test_cls_embs[i:i+args.cls_batch_size].to(args.device)
                y_logit = mlp_model(Xbatch)
                y_pred = torch.argmax(y_logit, -1)
                all_preds.extend(y_pred.cpu().numpy())
                all_labels.extend(test_labels[i:i+args.cls_batch_size].cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f'Epoch {epoch}: Loss {loss.item()}, Accuracy {accuracy}, F1 {f1}')

        epoch_accuracies.append(accuracy)
        epoch_losses.append(loss.item())
        f1_scores.append(f1)

    # Results
    print("Accuracies:", epoch_accuracies)
    print("f1 scores:", f1_scores)
    print("Losses:", epoch_losses)
    print("max accuracy:", max(epoch_accuracies))
    print("max f1:", max(f1_scores))

if __name__ == "__main__":
    args = get_args()
    tokenizer, base_model = get_pretrained_model(args)
    base_model = base_model.to(args.device)

    #load dataset for inference attack
    if args.attack_data == "tweets_gender":
        dataset = DatasetDict.load_from_disk('attack/tweets_gender')
        
    elif args.attack_data == "women_clothing":
        dataset = DatasetDict.load_from_disk('attack/women_clothing')
        
    ### load data
    train_inputs,train_labels = get_dataset(dataset, 'train',tokenizer,args)
    for key in train_inputs:
        train_inputs[key] = train_inputs[key][:1000]
    train_labels = train_labels[:1000]
    test_inputs,test_labels = get_dataset(dataset, 'test',tokenizer,args)

    # obtain the embeddings for clean embedding
    #train_cls_embs = get_embedding(train_inputs, base_model)
    #test_cls_embs = get_embedding(test_inputs, base_model)

    #inital attribute inference attack before privatization
    #print("original attribute inference accuracy:")
    #attribute_inference_attack(train_cls_embs, test_cls_embs)

    # Define noise levels
    args.test_eta = 0
 
    if "t5" in args.base_model:
        decoder_start_token = tokenizer("classify: ", return_tensors='pt').input_ids
        decoder_start_token = decoder_start_token.to(args.device)
    else:
        decoder_start_token = None
    
    # obtain the embeddings for privatization 
    # train_pr_inputs = text2text_priv(train_inputs, tokenizer, base_model, args)
    # test_pr_inputs = text2text_priv(test_inputs, tokenizer, base_model, args)
    train_token_embeddings, train_attn_masks = get_noisy_embedding(train_inputs, base_model, args)
    test_token_embeddings, test_attn_masks = get_noisy_embedding(test_inputs, base_model, args)

    train_cls_embs = get_embedding(train_token_embeddings, train_attn_masks, base_model, args, decoder_start_token)
    test_cls_embs = get_embedding(test_token_embeddings, test_attn_masks, base_model, args, decoder_start_token)
    
    base_mod_name = args.base_model.split('/')[-1]
    save_dir = os.path.join(args.denoise_data_dir, f"cls_embedding_{base_mod_name}_eta_{args.test_eta}")        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    torch.save(train_cls_embs,os.path.join(save_dir, 'train_cls_embs.pt'))
    torch.save(test_cls_embs,os.path.join(save_dir, 'test_cls_embs.pt'))
    torch.save(train_labels,os.path.join(save_dir, 'train_labels.pt'))
    torch.save(test_labels,os.path.join(save_dir, 'test_labels.pt'))

    print(f"\nTraining with differential privacy budget = {args.test_eta}")
    attribute_inference_attack(train_cls_embs, test_cls_embs, train_labels, test_labels)
