from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from models.model import *
from data.load_data import *
from denoise import test_denoise
import torch
from tqdm import tqdm
from util.parameters import get_args
import os
from data.dataset import DownStreamDataset
from torch.utils.data import DataLoader
from util.globals import *
from util.utils import get_cls_embedding, get_token_embedding, get_pretrained_model, get_finetuned_model, get_denoise_path
from huggingface_hub import login
login(token="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")

def get_embeddings_train_data(input_ids, attention_mask, base_model, denoise_model, args, decoder_start_token=None):
    '''
    server compute the embeddings given to the client
    '''
    #  get initial embeddings
    init_emb = get_token_embedding(input_ids, base_model, args)
            
    # sample noise
    if args.noise_mechanism == "Gaussian":
        noise = sample_noise_Gaussian(init_emb.shape, args.test_noise_std, args.device)
    elif args.noise_mechanism == "ChiDP":
        noise = sample_noise_Chi(init_emb.shape, args.test_eta, args.device)
        noise = noise.to(dtype=args.base_precision)

    noise_init_emb = init_emb + noise
    if args.clip == "element":
        lower, upper = emb_range_dict[args.base_model]
        noise_init_emb = torch.clamp(noise_init_emb, lower, upper)
        noise = noise_init_emb - init_emb
    elif args.clip == "norm":
        max_norm = emb_norm_dict[args.base_model]
        all_norms = torch.norm(noise_init_emb, p=2, dim=-1)
        noise_init_emb = noise_init_emb * torch.clamp(max_norm / all_norms, max=1).unsqueeze(-1)
        noise = noise_init_emb - init_emb
    
    with torch.no_grad():
        # get final embeddings without noise
        if "t5" in args.base_model:
            decoder_start_token = decoder_start_token.repeat(init_emb.shape[0], 1)
            clean_outputs = base_model(inputs_embeds=init_emb, attention_mask=attention_mask, 
                                       decoder_input_ids=decoder_start_token, output_hidden_states=True)
        else:
            clean_outputs = base_model(inputs_embeds=init_emb,  
                                        attention_mask=attention_mask, output_hidden_states=True)
        clean_cls_embs = get_cls_embedding(clean_outputs, attention_mask, args)

        # get final embeddings with noise
        if "t5" in args.base_model:
            noise_outputs = base_model(inputs_embeds=noise_init_emb, attention_mask=attention_mask, 
                                       decoder_input_ids=decoder_start_token, output_hidden_states=True)
        else:
            noise_outputs = base_model(inputs_embeds=noise_init_emb, attention_mask=attention_mask, output_hidden_states=True)
        noise_cls_embs = get_cls_embedding(noise_outputs, attention_mask, args)

        noise = noise.to(dtype=args.denoise_precision)
        init_emb = init_emb.to(dtype=args.denoise_precision)
        clean_cls_embs= clean_cls_embs.to(dtype=args.denoise_precision)
        noise_cls_embs = noise_cls_embs.to(dtype=args.denoise_precision)
        # denoise output
        attn_mask = attention_mask
        if args.mask_init:
            mask = attn_mask.unsqueeze(-1).expand_as(noise)
            noise = noise.masked_fill(mask == 0, 0)
            init_emb = init_emb.masked_fill(mask == 0, 0)
        if args.mask_attn:
            denoise_cls_emb = denoise_model(init_emb, noise, noise_cls_embs, attn_mask)
        else:
            denoise_cls_emb = denoise_model(init_emb, noise, noise_cls_embs)

    return denoise_cls_emb, noise_cls_embs, clean_cls_embs

def get_embeddings_test_data(input_ids, attention_mask, base_model, denoise_model, args, decoder_start_token=None):
    '''
    server compute the embeddings given to the client
    '''
    #  get initial embeddings
    init_emb = get_token_embedding(input_ids, base_model, args)
            
    # sample noise
    if args.noise_mechanism == "Gaussian":
        noise = sample_noise_Gaussian(init_emb.shape, args.test_noise_std, args.device)
    elif args.noise_mechanism == "ChiDP":
        noise = sample_noise_Chi(init_emb.shape, args.test_eta, args.device)
        noise = noise.to(dtype=args.base_precision)
    noise_init_emb = init_emb + noise
    if args.clip == "element":
        lower, upper = emb_range_dict[args.base_model]
        noise_init_emb = torch.clamp(noise_init_emb, lower, upper)
        noise = noise_init_emb - init_emb
    elif args.clip == "norm":
        max_norm = emb_norm_dict[args.base_model]
        all_norms = torch.norm(noise_init_emb, p=2, dim=-1)
        noise_init_emb = noise_init_emb * torch.clamp(max_norm / all_norms, max=1).unsqueeze(-1)
        noise = noise_init_emb - init_emb
    
    with torch.no_grad():
        # get final embeddings with noise
        if "t5" in args.base_model:
            decoder_start_token = decoder_start_token.repeat(init_emb.shape[0], 1)
            noise_outputs = base_model(inputs_embeds=noise_init_emb, attention_mask=attention_mask, 
                                       decoder_input_ids=decoder_start_token, output_hidden_states=True)
        else:
            noise_outputs = base_model(inputs_embeds=noise_init_emb, attention_mask=attention_mask, output_hidden_states=True)
        noise_cls_embs = get_cls_embedding(noise_outputs, attention_mask, args)

        noise = noise.to(dtype=args.denoise_precision)
        init_emb = init_emb.to(dtype=args.denoise_precision)
        noise_cls_embs = noise_cls_embs.to(dtype=args.denoise_precision)
        # denoise output
        attn_mask = attention_mask
        if args.mask_init:
            mask = attn_mask.unsqueeze(-1).expand_as(noise)
            noise = noise.masked_fill(mask == 0, 0)
            init_emb = init_emb.masked_fill(mask == 0, 0)
        if args.mask_attn:
            denoise_cls_emb = denoise_model(init_emb, noise, noise_cls_embs, attn_mask)
        else:
            denoise_cls_emb = denoise_model(init_emb, noise, noise_cls_embs)

    return denoise_cls_emb

def accuracy_test(args):
    if args.use_ft_base:
        tokenizer, base_model = get_finetuned_model(args)
    else:
        tokenizer, base_model = get_pretrained_model(args)
    
    base_model = base_model.to(args.device)
    ### load data
    # load train data
    train_dataset = DownStreamDataset('train', tokenizer, args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.base_batch_size, shuffle=False, num_workers=0, drop_last=True)
    print("length of training data: ", len(train_dataset))
    # # load test data
    test_dataset = DownStreamDataset('test', tokenizer, args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.base_batch_size, shuffle=False, num_workers=0, drop_last=True)
    print("length of testing data: ", len(test_dataset))
    # storage train labels and test labels for cls model
    train_labels = train_dataset.labels
    test_labels = test_dataset.labels
    print("train_labels: ", train_labels)
    print("test_labels: ", test_labels)
    print("train_label_device: ", train_labels.device)
    print("test_label_device: ", test_labels.device)
    emb_dim = emb_size_dict[args.base_model]
    n_labels = len(torch.unique(train_labels))

    max_sim = 0
    # Load the trained denoise model
    for d_epoch in range(args.denoise_epochs):
        print(f"Test for denoise model of epoch {d_epoch}")
        denoise_model = eval(args.denoise_model)(d_model=emb_dim, d_out=emb_dim, args=args)  # Set the required parameters
        model_path = get_denoise_path(args, d_epoch, args.use_ft_base)
        denoise_model.load_state_dict(torch.load(model_path, map_location=args.device))
        denoise_model.to(args.device)
        if "t5" in args.base_model:
            decoder_start_token = tokenizer("classify: ", return_tensors='pt').input_ids
            decoder_start_token = decoder_start_token.to(args.device)
        else:
            decoder_start_token = None
        noise_results = []
        noise_mse = 0
        denoise_mse = 0
        noise_sim = 0
        denoise_sim = 0
        denoise_train_cls_embs = torch.zeros(len(train_labels), emb_dim*train_dataset.n_keys)
        current_idx = 0 # 追踪当前写入的位置
        cnt = 0
        for batch in tqdm(train_dataloader):
            train_input_ids = batch['input_ids']
            train_att_masks = batch['attention_mask']
            for idx in range(train_dataset.n_keys):
                input_ids = train_input_ids[idx]
                att_masks = train_att_masks[idx]
                input_ids = input_ids.to(args.device)
                att_masks = att_masks.to(args.device)
                
                denoise_train_cls_emb, noise_cls_embs, clean_cls_emb = get_embeddings_train_data(input_ids, att_masks, base_model, denoise_model, args, decoder_start_token)
            # if torch.cuda.is_available():
            #     denoise_train_cls_emb = denoise_train_cls_emb.cpu()
                # torch.cuda.empty_cache()
                end_idx = current_idx + denoise_train_cls_emb.size(0)
                denoise_train_cls_embs[current_idx:end_idx, (idx*emb_dim):((idx+1)*emb_dim)] = denoise_train_cls_emb
                if idx == train_dataset.n_keys-1:
                    current_idx = end_idx
                
                # test denoise model performance
                this_noise_mse, this_denoise_mse, this_noise_sim, this_denoise_sim = test_denoise(clean_cls_emb, noise_cls_embs, denoise_train_cls_emb, args)
                noise_mse += this_noise_mse
                denoise_mse += this_denoise_mse
                noise_sim += this_noise_sim
                denoise_sim += this_denoise_sim
                if cnt % 100 == 0:
                    tqdm.write(f"MSE before denoise: {this_noise_mse}, MSE after denoise: {this_denoise_mse}")
                    tqdm.write(f"Cosine similarity before denoise: {this_noise_sim}, Cosine similarity after denoise: {this_denoise_sim}")
                cnt += 1
        print(f"Average MSE before denoise: {noise_mse/cnt}, Average MSE after denoise: {denoise_mse/cnt}")
        print(f"Average cosine similarity before denoise: {noise_sim/cnt}, Average cosine similarity after denoise: {denoise_sim/cnt}")

        
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
        print(f"Finish testing the MSE and cosine similarity for epoch {d_epoch}")
        
        # obtain testing embeddings and noises for cls model
        # n_samples = len(test_labels)
        denoise_test_cls_embs = torch.zeros(len(test_labels), emb_dim*train_dataset.n_keys)
        current_idx = 0  # 用于追踪当前写入位置
        for batch in tqdm(test_dataloader):
            test_input_ids = batch['input_ids']
            test_att_masks = batch['attention_mask']
            for idx in range(train_dataset.n_keys):
                input_ids = test_input_ids[idx]
                att_masks = test_att_masks[idx]
                input_ids = input_ids.to(args.device)
                att_masks = att_masks.to(args.device)
                
                denoise_test_cls_emb = get_embeddings_test_data(input_ids, att_masks, base_model, denoise_model, args, decoder_start_token)
                # if torch.cuda.is_available():
                #     denoise_test_cls_emb = denoise_test_cls_emb.cpu()
                    # torch.cuda.empty_cache()
                # denoise_test_cls_embs.append(denoise_test_cls_emb)
                end_idx = current_idx + denoise_test_cls_emb.size(0)
                denoise_test_cls_embs[current_idx:end_idx, (idx*emb_dim):((idx+1)*emb_dim)] = denoise_test_cls_emb
                if idx == train_dataset.n_keys-1:
                    current_idx = end_idx
        # denoise_test_cls_embs = torch.cat(denoise_test_cls_embs, dim=0)
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        avg_sim = denoise_sim/cnt
        if args.save_emb and avg_sim > max_sim:
            max_sim = avg_sim
            base_mod_name = args.base_model.split('/')[-1]
            if args.denoise_data == "mix":
                denoise_data_name = args.mixed_data_config_list.split(",")
                denoise_data_name = [i.strip()[0] for i in denoise_data_name]
                denoise_data_name = ''.join(denoise_data_name)
                save_dir = os.path.join(args.denoise_data_dir, f"embedding_{args.task}_{base_mod_name}_{args.denoise_data}_{denoise_data_name}_{args.denoise_model}_{args.comb}_{str(args.att_pool)}_{str(args.mask_init)}_{str(args.mask_attn)}")
            else:
                save_dir = os.path.join(args.denoise_data_dir, f"embedding_{args.task}_{base_mod_name}_{args.denoise_data}_{args.denoise_model}_{args.comb}_{str(args.att_pool)}_{str(args.mask_init)}_{str(args.mask_attn)}")        
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            torch.save(denoise_train_cls_embs,os.path.join(save_dir, 'denoise_train_cls_embs.pt'))
            torch.save(denoise_test_cls_embs,os.path.join(save_dir, 'denoise_test_cls_embs.pt'))
            torch.save(train_labels,os.path.join(save_dir, 'train_labels.pt'))
            torch.save(test_labels,os.path.join(save_dir, 'test_labels.pt'))
        # initialize the classifier (user side)
        input_dim = denoise_train_cls_embs.shape[-1]
        if args.task != 'glue_stsb':
            cls_model = EnhancedClsModel(input_dim, n_labels).to(args.device) # input_dim = 1600（GPT2 xl）, n_labels = 2（二分类）
            loss_fn = nn.CrossEntropyLoss()
        else:
            cls_model = regModel(input_dim).to(args.device)
            loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(cls_model.parameters(), lr=0.0001)

        # train the classifier (user side)
        epoch_accuracies = []
        epoch_losses = []
        epoch_aucs = []

        for epoch in range(args.cls_epochs):
            train_labels = train_labels.to(args.device)
            for i in tqdm(range(0, len(denoise_train_cls_embs), args.cls_batch_size)):
                cls_model.train()
                Xbatch = denoise_train_cls_embs[i:i+args.cls_batch_size]
                Xbatch = Xbatch.to(args.device)
                y_pred = cls_model(Xbatch)
                ybatch = train_labels[i:i+args.cls_batch_size]
                loss = loss_fn(y_pred, ybatch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # compute the prediction and accuracy (user side)
            y_preds = []
            with torch.no_grad():
                for i in tqdm(range(0, len(denoise_test_cls_embs), args.cls_batch_size)):
                    cls_model.eval()
                    Xbatch = denoise_test_cls_embs[i:i+args.cls_batch_size]
                    Xbatch = Xbatch.to(args.device)
                    if args.task != 'glue_stsb':
                        y_logit = cls_model(Xbatch)
                        y_pred = torch.argmax(y_logit, -1)
                    else:
                        y_pred = cls_model(Xbatch)
                    y_preds.append(y_pred)
            y_preds = torch.cat(y_preds)
            if 'cuda' in args.device:
                y_preds = y_preds.cpu()
                test_labels = test_labels.cpu()
            if args.task != 'glue_stsb':
                accuracy = accuracy_score(test_labels, y_preds)
                auc = roc_auc_score(test_labels, y_preds)
                epoch_aucs.append(auc)
                print(f'Finished epoch {epoch}, latest loss {loss}, accuracy {accuracy}, auc {auc}')
            else:
                accuracy = r2_score(test_labels, y_preds)
                print(f'Finished epoch {epoch}, latest loss {loss}, accuracy {accuracy}')

            # Add accuracy and loss to lists
            epoch_accuracies.append(accuracy)
            epoch_losses.append(loss.item())

        # Add results to overall results list
        if args.task != 'glue_stsb':
            noise_results.append({
                "epoch": d_epoch,
                "accuracies": epoch_accuracies,
                "losses": epoch_losses,
                "aucs": epoch_aucs,
            })

            max_auc = max(epoch_aucs)
            # max_idx = epoch_aucs.index(max_auc)
            max_acc = max(epoch_accuracies)
            print(f"Model {args.base_model} eta {args.test_eta}, max auc: {max_auc}, max accuracy: {max_acc}")


        # Print results
        print(f"Result for {args.base_model} eta {args.test_eta}:")
        for result in noise_results:
            print(result)

def test_syn(args):
    device = 'cuda'
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    size = 10  # or any other desired size
    inputs = synthesize_data(args.denoise_size, args.token_length, tokenizer, args.device)

    # Decode and print out the synthetic texts
    for idx, input_ids in enumerate(inputs['input_ids']):
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        print(f"Generated Text {idx + 1}: {text}")

if __name__ == "__main__":
    args = get_args()
    print(args)
    accuracy_test(args)