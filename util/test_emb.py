from util.utils import get_token_embedding, get_pretrained_model
from util.parameters import get_args
from data.dataset import UniversalDataset
from data.load_data import sample_noise_Chi
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

args = get_args()
tokenizer, base_model = get_pretrained_model(args)
# base_model = base_model.to(args.device)
train_dataset = UniversalDataset(args.denoise_data, 'train', tokenizer, args.denoise_data_percentage, args)
dataloader = DataLoader(train_dataset, batch_size=50, shuffle=False, num_workers=0, drop_last=True, pin_memory=True)
cosine_similarity = torch.nn.functional.cosine_similarity
for i, batch in enumerate(tqdm(dataloader)):
    # input_ids = batch['input_ids'].to(args.device)
    # attention_masks = batch['attention_mask'].to(args.device)
    input_ids = batch['input_ids'][:,:50]
    this_init_embs = get_token_embedding(input_ids, base_model, args)
    noises = sample_noise_Chi(this_init_embs.shape, 500, "cpu")
    noise_init_embs = this_init_embs+noises
    noise_sim = cosine_similarity(this_init_embs, noise_init_embs).mean()
    print("Average cosine similarity:", noise_sim)
    print("Average of init emb:", torch.mean(this_init_embs))
    print("1% of init emb:", torch.quantile(this_init_embs, 0.01))
    print("99% of init emb:", torch.quantile(this_init_embs, 0.99))
    print("99% of init emb norm:", torch.quantile(torch.norm(this_init_embs, p=2, dim=-1), 0.99))
    print("Standard deviation of init emb:", torch.std(this_init_embs))
    print("Standard deviation of noises:", torch.std(noises))
    print("1% of noisy init emb:", torch.quantile(noise_init_embs, 0.01))
    print("99% of noisy init emb:", torch.quantile(noise_init_embs, 0.99))
    print("99% of noisy init emb norm:", torch.quantile(torch.norm(noise_init_embs, p=2, dim=-1), 0.99))
    if i == 10:
        break