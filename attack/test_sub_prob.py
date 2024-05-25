from models.model import *
from util.parameters import get_args
from tqdm import tqdm
from util.utils import get_closest_token, get_token_embedding, get_pretrained_model

def sample_noise_Chi(size, eta, device):
    alpha = torch.ones(*size) * size[-1]
    beta = torch.ones(*size) * eta
    m = Gamma(alpha, beta)
    l_lst = m.sample()
    v_lst = -2 * torch.rand(*size) + 1
    noise = l_lst * v_lst
    noise = noise.to(device)
    return noise

# token embedding inversion attack
args = get_args()
tokenizer, model = get_pretrained_model(args)
model = model.to(args.device)
vocab_size = tokenizer.vocab_size  
n_samples = 1000
dp_levels = [0.1, 1, 10, 50, 100]  
for eta in dp_levels:
    print("Testing with differential privacy budget: ", eta)
    args.eta = eta
    sim_tokens = torch.randint(0, vocab_size, size=(n_samples,), device=args.device)
    init_embeddings = get_token_embedding(sim_tokens, model, args)
    # sample noise
    noises = sample_noise_Chi(init_embeddings.shape, args.eta, args.device)
    init_embeddings = init_embeddings + noises
    cnt = 0
    for i in tqdm(range(n_samples)):
        embed = init_embeddings[i]
        closest_token = get_closest_token(embed, tokenizer, model, args)
        this_token = sim_tokens[i].item()
        if this_token != closest_token:
            cnt += 1
    print(f"Substitution probability: {cnt/n_samples}")