from sklearn.metrics import r2_score, roc_auc_score, accuracy_score
from models.model import *
from data.load_data import *
from tqdm import tqdm
from util.parameters import get_args
from util.globals import *
from util.utils import get_pretrained_model, get_cls_embedding
import os
from data.dataset import DownStreamDataset
import numpy as np
import torch

args = get_args()
tokenizer, base_model = get_pretrained_model(args)
noise_results = []

def get_embedding(inputs, model, n_samples, decoder_start_token):
    '''
    server compute the embeddings given to the client
    '''
    cls_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, args.base_batch_size)):
            outputs = model(inputs['input_ids'][i:i+args.base_batch_size], attention_mask=inputs['attention_mask'][i:i+args.base_batch_size], decoder_start_token=decoder_start_token)
            clean_cls_embs = get_cls_embedding(outputs, inputs['attention_mask'][i:i+args.base_batch_size], args)
            if "cuda" in args.device:
                clean_cls_embs = clean_cls_embs.cpu()
            cls_embs.append(clean_cls_embs)
    cls_embs = torch.cat(cls_embs, 0)
    return cls_embs

### load data
# load train data
# train_inputs, train_labels, test_inputs, test_labels = get_dataset_universe(args.task, tokenizer, args)

def extract_inputs_from_dataset(dataset):
    n_keys = dataset.n_keys  # 从 DownStreamDataset 对象中获取 n_keys
    extracted_inputs = [{} for _ in range(n_keys)]

    for input_dict in dataset.inputs:
        for key_idx in range(n_keys):
            input_key = f'input_ids{key_idx + 1}'
            mask_key = f'attention_mask{key_idx + 1}'
            
            if 'input_ids' not in extracted_inputs[key_idx]:
                extracted_inputs[key_idx]['input_ids'] = []
            if 'attention_mask' not in extracted_inputs[key_idx]:
                extracted_inputs[key_idx]['attention_mask'] = []
            
            extracted_inputs[key_idx]['input_ids'].append(input_dict[input_key])
            extracted_inputs[key_idx]['attention_mask'].append(input_dict[mask_key])

    # 将列表转换为张量
    for key_idx in range(n_keys):
        extracted_inputs[key_idx]['input_ids'] = torch.stack(extracted_inputs[key_idx]['input_ids'])
        extracted_inputs[key_idx]['attention_mask'] = torch.stack(extracted_inputs[key_idx]['attention_mask'])

    return extracted_inputs

# 使用新函数
train_dataset = DownStreamDataset('train', tokenizer, args)
train_inputs = extract_inputs_from_dataset(train_dataset)
train_labels = train_dataset.labels

test_dataset = DownStreamDataset('test', tokenizer, args)
test_inputs = extract_inputs_from_dataset(test_dataset)
test_labels = test_dataset.labels


if args.task != 'glue_stsb':
    n_labels = len(torch.unique(train_labels))

# Define noise levels
dp_levels = [50, 100, 500]  
# bert-base/large: 50,100, 500,1000,2500,5000,6000,7500,9000,10000,0
# distillbert: 1,50,100,150,200,300,400,500,750,1000,0
# t5: 0.1,0.5,1,1.5,5,10,20,30,40,50,100,150,200,0
if "t5" in args.base_model:
    decoder_start_token = tokenizer("classify: ", return_tensors='pt').input_ids
    decoder_start_token = decoder_start_token.to(args.device)
else:
    decoder_start_token = None

for eta in dp_levels:
    print(f"\nTraining with differential privacy budget = {eta}")
    args.test_eta = eta
    if args.scaling:
        llm_model = ScaleModel(base_model, args).to(args.device)
    else:
        llm_model = NoisyModel(base_model, args).to(args.device)
    
    # obtain the embeddings (server side)
    train_cls_embs = []
    for inputs in train_inputs:
        train_cls_emb = get_embedding(inputs, llm_model, len(train_labels), decoder_start_token)
        train_cls_embs.append(train_cls_emb)
    train_cls_embs = torch.cat(train_cls_embs, dim=-1)
    test_cls_embs = []
    for inputs in test_inputs:
        test_cls_emb = get_embedding(inputs, llm_model, len(test_labels), decoder_start_token)
        test_cls_embs.append(test_cls_emb)
    test_cls_embs = torch.cat(test_cls_embs, dim=-1)
    
            
    # save clean embedding
    if eta == 0:
        base_mod_name = args.base_model.split('/')[-1]
        save_dir = os.path.join(args.denoise_data_dir, f"clean_cls_embedding_{args.task}_{base_mod_name}")        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        torch.save(train_cls_embs,os.path.join(save_dir, 'train_cls_embs.pt'))
        torch.save(test_cls_embs,os.path.join(save_dir, 'test_cls_embs.pt'))
        torch.save(train_labels,os.path.join(save_dir, 'train_labels.pt'))
        torch.save(test_labels,os.path.join(save_dir, 'test_labels.pt'))

    # fit the classifier with the embedding (user side)
    input_dim = train_cls_embs.shape[-1]
    if args.task != 'glue_stsb':
        cls_model = EnhancedClsModel(input_dim, n_labels).to(args.device)
        loss_fn = nn.CrossEntropyLoss()
    else:
        cls_model = regModel(input_dim).to(args.device)
        if args.loss == 'mse':
            loss_fn = nn.MSELoss()
        elif args.loss == 'cosEmb':
            loss_fn = nn.CosineEmbeddingLoss()
    optimizer = torch.optim.Adam(cls_model.parameters(), lr=0.001)

    epoch_accuracies = []
    epoch_losses = []
    epoch_aucs = []

    for epoch in range(args.cls_epochs):
        for i in tqdm(range(0, len(train_cls_embs), args.cls_batch_size)):
            Xbatch = train_cls_embs[i:i+args.cls_batch_size]
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
            for i in tqdm(range(0, len(test_cls_embs), args.cls_batch_size)):
                X_test = test_cls_embs[i:i+args.cls_batch_size]
                X_test = X_test.to(args.device)
                if args.task != 'glue_stsb':
                    y_logit = cls_model(X_test)
                    y_pred = y_logit.argmax(-1)
                else:
                    y_pred = cls_model(X_test)
                y_preds.append(y_pred)
        y_preds = torch.cat(y_preds)
        if 'cuda' in args.device:
            y_preds = y_preds.cpu()
            test_labels = test_labels.cpu()
        if args.task != 'glue_stsb':
            accuracy = accuracy_score(test_labels, y_preds)
            auc = roc_auc_score(test_labels, y_preds)
            print(f'Finished epoch {epoch}, latest loss {loss}, accuracy {accuracy}, auc {auc}')
            epoch_aucs.append(auc)
        else:
            accuracy = r2_score(test_labels, y_preds)
            print(f'Finished epoch {epoch}, latest loss {loss}, accuracy {accuracy}')        

        # Add accuracy and loss to lists
        epoch_accuracies.append(accuracy)
        epoch_losses.append(loss.item())

    # Add results to overall results list
    if args.task != 'glue_stsb':
        max_acc = max(epoch_accuracies)
        max_acc_index = epoch_accuracies.index(max_acc)
        max_auc = epoch_aucs[max_acc_index]
        this_rsl = {
            "dp_level": eta,
            "accuracies": epoch_accuracies,
            "aucs": epoch_aucs,
            "losses": epoch_losses,
            "max_acc": max_acc,
            "max_auc": max_auc,
            "mean_acc":sum(epoch_accuracies)/len(epoch_accuracies),
            "mean_auc":sum(epoch_aucs)/len(epoch_aucs)
        }
        
    else:
        this_rsl = {
            "dp_level": eta,
            "accuracies": epoch_accuracies,
            "losses": epoch_losses
        }
    noise_results.append(this_rsl)
    print(this_rsl)
print("Noises under different dp levels:")
print(noise_results)
# Initialize lists to store mean accuracy and loss values
mean_accuracies = []
mean_losses = []
dp_levels = []

for result in noise_results:
    print(result)

    # Calculate mean accuracy and loss and append to lists
    mean_accuracies.append(np.mean(result["accuracies"]))
    mean_losses.append(np.mean(result["losses"]))
    dp_levels.append(result['dp_level'])