import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

from util.globals import *

import json
import os
import codecs

# https://github.com/jonathandunn/corpus_similarity/tree/main

parent_dir = os.path.dirname(os.path.abspath(__file__))

def synthesize_data(size, token_length, tokenizer, args=None):
    # sample token and attention mask
    sim_lengths = torch.randint(1, token_length, size=(size,1)) # size = size
    vocab_size = tokenizer.vocab_size  #vocab_size = len(tokenizer)
    sim_tokens = torch.randint(0, vocab_size, size=(size, token_length))
    att_mask = torch.ones_like(sim_tokens)
    for i in range(size):
        this_len = sim_lengths[i]
        sim_tokens[i, this_len:] = 0
        att_mask[i, this_len:] = 0
    return {'input_ids': sim_tokens, 'attention_mask': att_mask}

def get_dataset_options(d_name, size, args=None):
    if d_name not in dataset_configs:
        raise ValueError(f"Unknown dataset name: {d_name}")

    config = dataset_configs[d_name]
    dataset = load_dataset(config['dataset'], config['subset'], split='train') if config['subset'] else load_dataset(config['dataset'], split='train')
    # print(len(dataset))
    subset_indices = np.random.choice(len(dataset), min(size, len(dataset)), replace=False)
    sample_dataset = dataset.select(subset_indices)
    if isinstance(config['key'], list):
        text = []
        for key in config['key']:
            cur_text = sample_dataset[key]
            text.extend(cur_text)
    else:
        text = sample_dataset[config['key']]
    if d_name == 'daily_dialog':
        temp = []
        for t in text:
            t = " ".join(t)
            temp.append(t)
        text = temp
    return text

def get_features(vectorizer, lines):

    X = vectorizer.transform(lines)  
    fre_array = X.toarray()
    fre_array_sum = np.sum(fre_array, axis=0)

    return fre_array_sum

def calculate(lines1, lines2, vectorizer):

    features1 = get_features(vectorizer, lines1)
    features2 = get_features(vectorizer, lines2)

    value = spearmanr(features1, features2)[0]

    return value

if __name__ == "__main__":
    SAMPLE_SIZE = 20000
    TOKEN_LENGTH = 512
    feature_file = os.path.join(parent_dir, "eng_5k_OUT_char4.json")
    with codecs.open(feature_file, "r", encoding = "utf-8") as fo:
        feature_set = json.load(fo)
        feature_set = list(feature_set["eng"].values())
    
    vectorizer = CountVectorizer(analyzer = 'char_wb', ngram_range = (4, 4), vocabulary = feature_set)
    public_datasets = ['wikitext', "webtext", 'squad', 'ag_news', 'phrasebank', 
                   'banking77', 'health_fact', 'poem_sentiment', 'tweet_sentiment', 'tweet_emotion', 
                    'tweet_hate', 'tweet_offensive', 'ade_corpus_v2', 'hate_speech18', 'sms_spam', 
                    'yelp_review_full', 'app_reviews', 'rotten_tomatoes', 'daily_dialog']
    private_datasets = ["glue_mrpc", 'glue_rte', 'glue_cola']
    # compute the similarity between public and private dataset
    for private_data in private_datasets:
        private_lines = get_dataset_options(private_data, SAMPLE_SIZE)
        for public_data in public_datasets:
            print(f"Calculating the similarity between {private_data} and {public_data}")
            public_lines = get_dataset_options(public_data, SAMPLE_SIZE)
            sim = calculate(private_lines, public_lines, vectorizer)
            print(f"Similarity between {private_data} and {public_data} is {sim}")
    
    # compute the similarity between synthetic and private dataset
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    syn_inputs = synthesize_data(SAMPLE_SIZE, TOKEN_LENGTH, tokenizer)
    syn_sents = tokenizer.batch_decode(syn_inputs['input_ids'], skip_special_tokens=True)
    for private_data in private_datasets:
        print(f"Calculating the similarity between {private_data} and synthetic data")
        private_lines = get_dataset_options(private_data, SAMPLE_SIZE)
        sim = calculate(private_lines, syn_sents, vectorizer)
        print(f"Similarity between {private_data} and synthetic data is {sim}")