import torch
import numpy as np
from scipy.spatial.distance import cdist

from util.utils import get_pretrained_model,  get_token_embedding
from util.parameters import get_args
from util.globals import *

from data.load_data import sample_noise_Chi


def kth_nearest_neighbor_distances(X, k=1):
    """Returns the distance for the kth nearest neighbor of each point.
    """
    nX = len(X)
    # make sure X has the right shape for the cdist function
    X = np.reshape(X, (nX,-1))
    dists_arr = cdist(X, X)
    # sorts each row
    dists_arr.sort()
    return [dists_arr[i][k] for i in range(nX)]

if __name__ == "__main__":
    args = get_args()
    models = ["gpt2", "gpt2-medium","gpt2-large", "gpt2-xl"]
    # models = ["t5-small", "t5-base", "t5-large"]
    # models = ["stevhliu/my_awesome_model", "bert-base-uncased", "bert-large-uncased"]
    for model in models:
        args.base_model = model
        tokenizer, model = get_pretrained_model(args)
        vocab_size = tokenizer.vocab_size 
        etas = [1, 50, 100, 500, 750, 1000, 1250, 1500] 
        # etas = [0.1, 1, 10, 50, 100]
        # etas = [50, 100, 500, 750, 1000, 1250, 1500]
        # print(vocab_size)
        N_SAMPLES = 5000
        K = 1
        for eta in etas:
            print(f"Computing entropy for model {args.base_model} eta {eta}")
            # get token embeddings
            sim_tokens = torch.randperm(vocab_size)[:N_SAMPLES]
            init_embeddings = get_token_embedding(sim_tokens, model, args)
            # get noises
            noises = sample_noise_Chi(init_embeddings.shape, eta)
            noisy_init_embeddings = init_embeddings + noises
            # compute proxy of entropy for noisy embeddings
            dim = init_embeddings.shape[-1]
            r_k = kth_nearest_neighbor_distances(noisy_init_embeddings, k=K)
            l_k = np.log(r_k)
            Y_entropy = np.sum(l_k) * dim / N_SAMPLES
            # compute proxy of entropy for noise
            r_k = kth_nearest_neighbor_distances(noises, k=K)
            l_k = np.log(r_k)
            Z_entropy = np.sum(l_k) * dim / N_SAMPLES
            # obtain mutual information
            print(f"Mutual information for model {args.base_model} with eta {eta} is: {Y_entropy - Z_entropy}")