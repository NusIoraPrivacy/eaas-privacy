from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from models.model import *
from data.load_data import *
from tqdm import tqdm
from util.parameters import get_args
from util.globals import *
from util.utils import text2text_priv, get_pretrained_model, get_cls_embedding
import os
from torch import nn
import torch

args = get_args()
tokenizer, base_model = get_pretrained_model(args)
base_model = base_model.to(args.device)
#llm_model = NoisyModel(base_model).to(device)
# Store accuracy and loss for each noise level and epoch
noise_results = []

def get_embedding(inputs, model, n_samples, decoder_start_token):
    '''
    server compute the embeddings given to the client
    '''
    cls_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, args.base_batch_size)):
            input_ids = inputs['input_ids'][i:i+args.base_batch_size]
            att_masks = inputs['attention_mask'][i:i+args.base_batch_size]
            if "t5" in args.base_model:
                this_decoder_start_token = decoder_start_token.repeat(input_ids.shape[0], 1)
                outputs = model(input_ids=input_ids, 
                                attention_mask=att_masks, 
                                output_hidden_states=True, decoder_input_ids=this_decoder_start_token)
            else:
                outputs = model(input_ids=input_ids, 
                                attention_mask=att_masks, 
                                output_hidden_states=True)
            clean_cls_embs = get_cls_embedding(outputs, inputs['attention_mask'][i:i+args.base_batch_size], args)
            if "cuda" in args.device:
                clean_cls_embs = clean_cls_embs.cpu()
            cls_embs.append(clean_cls_embs)
    cls_embs = torch.cat(cls_embs, 0)
    return cls_embs

### load data
# load train data
raw_train_inputs, train_labels, raw_test_inputs, test_labels = get_dataset_universe(args.task, tokenizer, args)

if args.task != 'glue_stsb':
    n_labels = len(torch.unique(train_labels))

# sample data
if args.task == 'glue_qqp':
    selected_indices = np.random.choice(len(train_labels), 
                                        min(args.downstream_task_train_size, len(train_labels)), replace=False)
    temp = []
    for inputs in raw_train_inputs:
        for key in inputs:
            inputs[key] = inputs[key][selected_indices]
        temp.append(inputs)
    raw_train_inputs = temp
    train_labels = train_labels[selected_indices]

    selected_indices = np.random.choice(len(test_labels), 
                                        min(args.downstream_task_test_size, len(test_labels)), replace=False)
    temp = []
    for inputs in raw_test_inputs:
        for key in inputs:
            inputs[key] = inputs[key][selected_indices]
        temp.append(inputs)
    raw_test_inputs = temp
    test_labels = test_labels[selected_indices]

# Define noise levels
dp_levels = [0, 50, 100, 500]
if args.task == 'glue_qqp':
    assert dp_levels[0] == 0
if "t5" in args.base_model:
    decoder_start_token = tokenizer("classify: ", return_tensors='pt').input_ids
    decoder_start_token = decoder_start_token.to(args.device)
else:
    decoder_start_token = None
for eta in dp_levels:
    print(f"\nTraining with differential privacy budget = {eta}")
    args.test_eta = eta
    train_inputs = []
    for inputs in raw_train_inputs:
        if eta != 0:
            inputs = text2text_priv(inputs, tokenizer, base_model, args)
        train_inputs.append(inputs)

    test_inputs = []
    for inputs in raw_test_inputs:
        if eta != 0:
            inputs = text2text_priv(inputs, tokenizer, base_model, args)
        test_inputs.append(inputs)
    
    # obtain the embeddings (server side)
    train_cls_embs = []
    for inputs in train_inputs:
        train_cls_emb = get_embedding(inputs, base_model, len(train_labels), decoder_start_token)
        train_cls_embs.append(train_cls_emb)
    train_cls_embs = torch.cat(train_cls_embs, dim=-1)
    test_cls_embs = []
    for inputs in test_inputs:
        test_cls_emb = get_embedding(inputs, base_model, len(test_labels), decoder_start_token)
        test_cls_embs.append(test_cls_emb)
    test_cls_embs = torch.cat(test_cls_embs, dim=-1)

    # save clean embedding
    base_mod_name = args.base_model.split('/')[-1]
    if eta == 0:
        save_dir = os.path.join(args.denoise_data_dir, f"clean_cls_embedding_{args.task}_{base_mod_name}")
    else:
        save_dir = os.path.join(args.denoise_data_dir, f"clean_cls_embedding_{args.task}_{base_mod_name}_eta{eta}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    torch.save(train_cls_embs,os.path.join(save_dir, 'train_cls_embs.pt'))
    torch.save(test_cls_embs,os.path.join(save_dir, 'test_cls_embs.pt'))
    torch.save(train_labels,os.path.join(save_dir, 'train_labels.pt'))
    torch.save(test_labels,os.path.join(save_dir, 'test_labels.pt'))

    # compute cosine similarity and mse
    clean_dir = os.path.join(args.denoise_data_dir, f"clean_cls_embedding_{args.task}_{base_mod_name}")
    if os.path.exists(clean_dir):
        clean_train_cls_embs = torch.load(os.path.join(clean_dir, 'train_cls_embs.pt'))
        cosine_similarity = torch.nn.functional.cosine_similarity
        mse_metric = nn.MSELoss()
        # compute mse
        mse = mse_metric(clean_train_cls_embs, train_cls_embs)
        # compute cosine similarity
        cos_sim = cosine_similarity(clean_train_cls_embs, train_cls_embs).mean()
        print(f"MSE: {mse}, cosine similarity: {cos_sim}")

    # fit the classifier with the embedding (user side)
    input_dim = train_cls_embs.shape[-1]
    if args.task != 'glue_stsb':
        cls_model = clsModel(input_dim, n_labels).to(args.device)
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