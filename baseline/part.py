from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from models.model import *
from data.load_data import *
from tqdm import tqdm
from util.parameters import get_args
from util.globals import *
from util.utils import get_pretrained_model
from data.dataset import DownStreamDatasetSep
from torch.utils.data import DataLoader
import os
import torch
from transformers import get_linear_schedule_with_warmup

args = get_args()
tokenizer, base_model = get_pretrained_model(args)
base_model = base_model.to(args.device)

# load train data
train_dataset = DownStreamDatasetSep('train', tokenizer, base_model, args)
train_dataloader = DataLoader(train_dataset, batch_size=args.base_batch_size, shuffle=False, num_workers=0, drop_last=False)
print("length of training data: ", len(train_dataset))
# load test data
test_dataset = DownStreamDatasetSep('test', tokenizer, base_model, args)
test_dataloader = DataLoader(test_dataset, batch_size=args.base_batch_size, shuffle=False, num_workers=0, drop_last=False)

# get prompt tuning model
peft_model = PARTModel(base_model, train_dataset.n_labels, args)
peft_model = peft_model.to(args.device)
for param in peft_model.base_model.parameters():
    param.requires_grad = False

total_params  = sum(p.numel() for p in peft_model.parameters())
print("total parameters: ", total_params)
total_params  = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
print("total trainable parameters: ", total_params)

optimizer = torch.optim.AdamW(peft_model.parameters(), lr=args.lr_peft)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * args.train_epochs),
)

if "t5" in args.base_model:
    decoder_start_token = tokenizer("classify: ", return_tensors='pt').input_ids
    decoder_start_token = decoder_start_token.to(args.device)

# train PART model
epoch_accuracies = []
epoch_losses = []
epoch_aucs = []
for epoch in range(args.train_epochs):
    for batch in tqdm(train_dataloader):
        train_input_ids = batch['input_ids']
        train_att_masks = batch['attention_mask']
        train_labels = batch['labels']
        plain_tokens = batch['plain_token']
        if "t5" in args.base_model:
            loss = peft_model(input_ids=train_input_ids, 
                                plain_tokens=plain_tokens, 
                                attention_mask=train_att_masks, 
                                labels=train_labels, 
                                decoder_start_token=decoder_start_token,
                                )
        else:
            loss = peft_model(input_ids=train_input_ids, 
                                plain_tokens=plain_tokens, 
                                attention_mask=train_att_masks, 
                                labels=train_labels,
                                )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    # test PART model
    y_preds = []
    all_test_labels = []
    for batch in tqdm(test_dataloader):
        test_input_ids = batch['input_ids']
        test_att_masks = batch['attention_mask']
        test_labels = batch['labels']
        plain_tokens = batch['plain_token']
        if "t5" in args.base_model:
            y_logit = peft_model(input_ids=test_input_ids, 
                                plain_tokens=plain_tokens, 
                                attention_mask=test_att_masks, 
                                decoder_start_token=decoder_start_token,
                                )
        else:
            y_logit = peft_model(input_ids=test_input_ids, 
                                plain_tokens=plain_tokens, 
                                attention_mask=test_att_masks, 
                                )
        
        y_pred = torch.argmax(y_logit, -1)
        y_preds.append(y_pred)
        all_test_labels.append(test_labels)
    y_preds = torch.cat(y_preds)
    all_test_labels = torch.cat(all_test_labels)
    if 'cuda' in args.device:
        y_preds = y_preds.cpu()
        all_test_labels = all_test_labels.cpu()
    if args.task != 'glue_stsb':
        accuracy = accuracy_score(all_test_labels, y_preds)
        auc = roc_auc_score(all_test_labels, y_preds)
        epoch_aucs.append(auc)
        epoch_accuracies.append(accuracy)
        print(f'Finished epoch {epoch} for eta {args.test_eta} model {args.base_model}, latest loss {loss.item()}, accuracy {accuracy}, auc {auc}')

if args.task != 'glue_stsb':
    noise_results = {
        "accuracies": epoch_accuracies,
        "losses": epoch_losses,
        "aucs": epoch_aucs,
    }
    print(f"Result for eta {args.test_eta} model {args.base_model}")
    print(noise_results)