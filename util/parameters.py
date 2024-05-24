import argparse
# import sys
# import os
# current = os.path.dirname(os.path.realpath(__file__))
# parent = os.path.dirname(current)
# sys.path.append(parent)
from util.utils import str2bool, str2type
import torch

def get_args():
    # python

    parser = argparse.ArgumentParser()
    parser.add_argument("--denoise_epochs", type=int, default=2, 
                        help = "number of epochs to train denoise model")
    parser.add_argument("--cls_epochs", type=int, default=500, 
                        help = "number of epochs to train classification model")
    parser.add_argument("--train_epochs", type=int, default=4, 
                        help = "number of epochs to prompt-tuning base model for PART")
    parser.add_argument("--base_batch_size", type=int, default=6, 
                        help = "batch size to make prediction with LLM")
    parser.add_argument("--ft_batch_size", type=int, default=6, 
                        help = "batch size to finetune LLM")
    parser.add_argument("--device", type=str, default="cuda:0", 
                        choices = ["cuda", "cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                        help = "cpu or gpu")
    parser.add_argument("--cls_batch_size", type=int, default=512, 
                        help = "batch size to train classification model")
    parser.add_argument("--base_model", type=str, default="stevhliu/my_awesome_model", 
                        choices=["stevhliu/my_awesome_model", "gpt2", "gpt2-xl",
                                 "facebook/opt-2.7b","facebook/opt-6.7b","facebook/opt-350m",'facebook/opt-125m',
                                 "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf", "bert-base-uncased", "bert-large-uncased",
                                 "t5-small", "t5-base", "t5-large","gpt2-medium","gpt2-large"],
                        help = "large language model for prediction")
    parser.add_argument("--task", type=str, default="glue_mrpc", 
                        choices=["imdb", "glue_sst2",'glue_qqp','glue_stsb','glue_rte', 'glue_cola', 'glue_mrpc'],
                        help = "downstream task")
    parser.add_argument("--downstream_task_train_size", type=int, default=5000,
                        help = "size of data to train the downstream_task model")
    parser.add_argument("--downstream_task_test_size", type=int, default=1000,
                        help = "size of data to test the downstream_task model")
    parser.add_argument("--llama_dir", type=str, 
                        help = "directory of LLaMa")
    parser.add_argument("--denoise_data_dir", type=str, default="/home/hzyr/llm/denoise/data",
                        help="directory to store data that trains the denoise model")
    parser.add_argument("--denoise_size", type=int, default=20000,
                        help = "size of data to train the denoise model")
    parser.add_argument("--denoise_batch_size", type=int, default=12,
                        help = "batch size of data to train the denoise model")
    parser.add_argument("--train_noise_std", type=float, default=2.0,
                        help ="standard deviation of the noise to train the denoise model")
    parser.add_argument("--test_noise_std", type=float, default=2.0,
                        help ="standard deviation of the noise to test the denoise model")
    parser.add_argument("--train_eta", type=float, default=100,
                        help ="differential privacy budget to train the denoise model")
    parser.add_argument("--test_eta", type=float, default=100,
                        help ="differential privacy budget to test the denoise model")
    parser.add_argument("--noise_mechanism", type=str, default="ChiDP",
                        choices = ["Gaussian", "ChiDP"],
                        help ="noise mechanism")
    parser.add_argument("--denoise_data", type=str, default="mix",
                        choices=["imdb",'wikitext', "webtext", "synthetic", "mix", 'squad', 'glue_sst2', 'glue_cola', 
                        'ag_news', 'phrasebank', 'banking77', 'health_fact', 'poem_sentiment', 'tweet_sentiment', 'tweet_emotion', 
                        'tweet_hate', 'tweet_offensive', 'ade_corpus_v2', 'hate_speech18', 'sms_spam', 
                        'yelp_review_full', 'app_reviews', 'amazon_polarity', 'rotten_tomatoes', 'daily_dialog'],
                        help = "data set to train the denoise model")
    parser.add_argument("--ft_data", type=str, default="glue_sst2",
                        choices=["imdb",'wikitext', "webtext", "synthetic", 'squad', 'glue_sst2', 'glue_cola', 
                        'ag_news', 'phrasebank', 'banking77', 'health_fact', 'poem_sentiment', 'tweet_sentiment', 'tweet_emotion', 
                        'tweet_hate', 'tweet_offensive', 'ade_corpus_v2', 'hate_speech18', 'sms_spam', 
                        'yelp_review_full', 'app_reviews', 'amazon_polarity', 'rotten_tomatoes', 'daily_dialog'],
                        help = "data set to train the denoise model")
    parser.add_argument("--use_ft_base", type=str2bool, default=False,
                        help = "whether to use the finetuned base model or not")
    parser.add_argument("--ft_denoise_sample_size", type=int, default=100,
                        help = "sample size to finetune denoise model")
    parser.add_argument("--mixed_data_config_list", type=str, 
                        default=("tweet_offensive:train:5000, daily_dialog:train:5000, hate_speech18:train:5000, health_fact:train:5000, "
                        "squad:train:5000, ag_news:train:5000, phrasebank:train:5000, banking77:train:5000, "
                        "poem_sentiment:train:5000, tweet_sentiment:train:5000, tweet_emotion:train:5000, tweet_hate:train:5000, "
                        "ade_corpus_v2:train:5000, sms_spam:train:5000, yelp_review_full:train:5000, app_reviews:train:5000, "
                        "amazon_polarity:train:5000, rotten_tomatoes:train:5000, wikitext:train:5000, webtext:train:5000"),
                        help="List of datasets to mix in the format name:mode:sample_size separated by commas")
    # parser.add_argument("--mixed_data_config_list", type=str, 
    #                     default=("tweet_offensive:train:5000, daily_dialog:train:5000, hate_speech18:train:5000, health_fact:train:5000, "
    #                     "squad:train:5000, ag_news:train:5000, phrasebank:train:5000, banking77:train:5000"),
    #                     help="List of datasets to mix in the format name:mode:sample_size separated by commas")
    parser.add_argument("--denoise_model_dir", type=str, default="/home/hzyr/llm/denoise/model",
                        help = "directory to store the denoise model")
    parser.add_argument("--token_length", type=int, default=512,
                        help = "length of token sequence")
    parser.add_argument("--att_pool", type=str2bool, default=True,
                        help = "use attention pooling to combine sequence in the transformer")
    parser.add_argument("--comb", type=str, default="att_w_output_v3",
                        choices=["MLP", "attention", "att_w_output", "att_w_output_v2", "MLP_v2",
                                 "MLP_v3", "att_w_output_v3", "att_att_v1", "select_att_v1"],
                        help = "combine representation from noise, embedding, and purturbed output")
    parser.add_argument("--num_heads", type=int, default=8,
                        help = "number of heads in the transformer")
    parser.add_argument("--n_emb_block", type=int, default=1,
                        help = "number of transformer blocks for token embedding (recommend to have 1 block for user computation efficiency)")
    parser.add_argument("--n_noise_block", type=int, default=1,
                        help = "number of transformer blocks for noise matrix (recommend to have 1 block for user computation efficiency)")
    parser.add_argument("--num_layers", type=int, default=6,
                        help = "number of layers in the transformer")
    parser.add_argument("--d_ff", type=int, default=1280,
                        help = "hidden dimension for the positional feed forward network")
    parser.add_argument("--dim_head", type=int, default=256,
                        help = "dimension of each head")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help = "dropout rate of the denoise model")
    parser.add_argument("--loss", type=str, default="mse",
                        choices = ["mse", "cosEmb"],
                        help ="type of training loss")
    parser.add_argument("--noise_per_sample", type=int, default=2,
                        help = "number of noise matrices added to each sample")
    parser.add_argument("--mask_init", type=str2bool, default=True,
                        help = "whether to mask the initial embedding and noise matrix")
    parser.add_argument("--mask_attn", type=str2bool, default=True,
                        help = "whether to have mask in the attention layer") # at least one of mask_init and mask_attn has to be true
    parser.add_argument("--denoise_model", type=str, default="denoiseModelv3",
                        choices=["denoiseModel", "denoiseModelv2", "denoiseModelCrsAtt", 
                                 "denoiseModelCrsAttv2", "denoiseModelv3"],
                        help = "denoise model")
    parser.add_argument("--gen_data", type=str2bool, default=False,
                        help = "whether to generate the embeddings and noises from text data")
    parser.add_argument("--base_precision", type=str2type, default=torch.float32,
                        help = "Precision of base model")
    parser.add_argument("--denoise_precision", type=str2type, default=torch.float32,
                        help = "Precision of base model")
    parser.add_argument("--train_sync", type=str2bool, default=True,
                        help = "Whether to combine the training and data generation process for denoise model")
    parser.add_argument("--save_emb", type=str2bool, default=True,
                        help = "Whether to save the denoised embedding")
    parser.add_argument("--denoise_data_percentage", type=float, default=0.5,
                        help = "Percentage of imdb test data to sample")
    parser.add_argument("--ft_data_size", type=float, default=1000,
                        help = "Size of finetuning data to sample")
    parser.add_argument("--ckpt_path", type=str,
                        help = "Path to the checkpoint of denoise model")
    parser.add_argument("--clip", type=str, default="norm",
                        choices=["none", "element", "norm"],
                        help = "Whether to clip the embeddings or not")
    parser.add_argument("--num_virtual_tokens", type=int, default=150,
                        help = "number of virtual tokens in prompt tuning")
    parser.add_argument("--lr_peft", type=float, default=6e-5,
                        help = "learning rate in prompt tuning")
    parser.add_argument("--n_plain_tok", type=int, default=40,
                        help = "number of plain text token for reconstruction")
    parser.add_argument("--rec_vocab_size", type=int, default=7630,
                        help = "vocabulary size of reconstruction layer")
    parser.add_argument("--attack_data", type=str, default="women_clothing",
                        choices=["tweets_gender", "women_clothing"],
                        help = "dataset for inference attack")
    parser.add_argument("--time_steps", type=int, default=5,
                        help = "number of steps to compute the training time of denoise model")
    parser.add_argument("--cal_time", type=str2bool, default=False,
                        help = "whether to compute the training time of a model")
    parser.add_argument("--scaling", type=str2bool, default=False,
                        help = "whether to use scaling for denoise")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    mixed_data = args.mixed_data_config_list.split(",")
    mixed_data = [i.strip()[0] for i in mixed_data]
    mixed_data = ''.join(mixed_data)
    print(mixed_data)
    print(str(args.att_pool))