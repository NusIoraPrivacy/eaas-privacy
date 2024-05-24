emb_size_dict = {
    'stevhliu/my_awesome_model': 768,
    "gpt2": 768,
    "gpt2-large": 1280,
    "gpt2-medium": 1024,
    "gpt2-xl": 1600,
    "facebook/opt-125m":768,
    "facebook/opt-350m":512,
    "ArthurZ/opt-350m-dummy-sc":512,
    "facebook/opt-2.7b":2560,
    "facebook/opt-6.7b":4096,
    "meta-llama/Llama-2-7b-hf":4096,
    "meta-llama/Llama-2-7b-chat-hf":4096,
    "bert-base-uncased":768,
    "bert-large-uncased":1024,
    "t5-small": 512,
    "t5-base":768,
    "t5-large":1024,
        }

emb_norm_dict = {
    'stevhliu/my_awesome_model': 2.5,
    "gpt2": 4,
    "gpt2-large": 2.2,
    "gpt2-medium": 4,
    "gpt2-xl": 2,
    "facebook/opt-125m": 0,
    "facebook/opt-350m": 0,
    "ArthurZ/opt-350m-dummy-sc": 0,
    "facebook/opt-2.7b": 0,
    "facebook/opt-6.7b": 0,
    "meta-llama/Llama-2-7b-hf": 0,
    "meta-llama/Llama-2-7b-chat-hf": 0,
    "bert-base-uncased": 2.5,
    "bert-large-uncased": 2.5,
    "t5-small": 850,
    "t5-base": 680,
    "t5-large": 650,
        }

emb_range_dict = {
    'stevhliu/my_awesome_model': (-0.15, 0.18),
    "gpt2": (-0.3, 0.28),
    "gpt2-large": (-0.1, 0.1),
    "gpt2-medium": (-0.2, 0.23),
    "gpt2-xl": (-0.09, 0.08),
    "facebook/opt-125m":(),
    "facebook/opt-350m": (),
    "ArthurZ/opt-350m-dummy-sc": (),
    "facebook/opt-2.7b": (),
    "facebook/opt-6.7b": (),
    "meta-llama/Llama-2-7b-hf": (),
    "meta-llama/Llama-2-7b-chat-hf": (),
    "bert-base-uncased": (-0.1, 0.15),
    "bert-large-uncased": (-0.1, 0.15),
    "t5-small": (-45, 38),
    "t5-base": (-38, 28),
    "t5-large": (-28, 25),
        }

ft_dataset_configs = {
        'glue_sst2': {'dataset': 'glue', 'subset': 'sst2', 'key': 'sentence', 'label': 'label', 'train': 'train', 'test':'validation'},
    }

task_dataset_configs = {
        'glue_sst2': {'dataset': 'glue', 'subset': 'sst2', 'key': ['sentence'],  'train': 'train', 'test':'validation'},#sentiment classification
        'glue_qqp': {'dataset': 'glue', 'subset': 'qqp', 'key': ['question1', 'question2'], 'train': 'train', 'test':'validation'},
        'imdb': {'dataset': 'imdb', 'subset': None, 'key': ['text'], 'train': 'train', 'test': 'test'},
        'glue_stsb': {'dataset': 'glue', 'subset': 'stsb', 'key': ['sentence1', 'sentence2'], 'train': 'train', 'test':'validation'},
        'glue_rte': {'dataset': 'glue', 'subset': 'rte', 'key': ['sentence1', 'sentence2'], 'train': 'train', 'test':'validation'},
        'glue_cola': {'dataset': 'glue', 'subset': 'cola', 'key': ['sentence'],  'train': 'train', 'test':'validation'},
        'glue_mrpc': {'dataset': 'glue', 'subset': 'mrpc', 'key': ['sentence1', 'sentence2'],  'train': 'train', 'test':'validation'},
    }

dataset_configs = {
        'squad': {'dataset': 'rajpurkar/squad', 'subset': None, 'key': 'question'},
        'glue_sst2': {'dataset': 'glue', 'subset': 'sst2', 'key': 'sentence'},#sentiment classification
        'glue_cola': {'dataset': 'glue', 'subset': 'cola', 'key': 'sentence'},#grammer checking
        'ag_news': {'dataset': 'ag_news', 'subset': None, 'key': 'text'},  # multiclass-sports/business/tech
        'phrasebank': {'dataset': 'financial_phrasebank', 'subset': 'sentences_allagree', 'key': 'sentence'}, # 3-class sentiments in financial domain
        'banking77': {'dataset': 'PolyAI/banking77', 'subset': None, 'key': 'text'}, # online banking queries annotated with their corresponding intents
        'health_fact': {'dataset': 'health_fact', 'subset': None, 'key': 'main_text'},# explainable automated fact-checking of public health claims.
        'poem_sentiment': {'dataset': 'poem_sentiment', 'subset': None, 'key': 'verse_text'},
        'tweet_sentiment': {'dataset': 'tweet_eval', 'subset': 'sentiment', 'key': 'text'},
        'tweet_emotion': {'dataset': 'tweet_eval', 'subset': 'emotion', 'key': 'text'}, 
        'tweet_hate': {'dataset': 'tweet_eval', 'subset': 'hate', 'key': 'text'},
        'tweet_offensive': {'dataset': 'tweet_eval', 'subset': 'offensive', 'key': 'text'}, 
        'ade_corpus_v2': {'dataset': 'ade_corpus_v2', 'subset': 'Ade_corpus_v2_classification', 'key': 'text'},# Classify if a sentence is Adverse Drug Reaction Data-related (True) or not (False) 
        'hate_speech18': {'dataset': 'hate_speech18', 'subset': None, 'key': 'text'},
        'sms_spam': {'dataset': 'sms_spam', 'subset': None, 'key': 'sms'},
        'daily_dialog': {'dataset': 'daily_dialog', 'subset': None, 'key': 'dialog'},
        # datasets below are about reviews/recommendations
        'yelp_review_full': {'dataset': 'yelp_review_full', 'subset': None, 'key': 'text'}, 
        'app_reviews': {'dataset': 'app_reviews', 'subset': None, 'key': 'review'},
        'amazon_polarity': {'dataset': 'amazon_polarity', 'subset': None, 'key': 'content'},
        'rotten_tomatoes': {'dataset': 'rotten_tomatoes', 'subset': None, 'key': 'text'},
        'wikitext': {'dataset': 'wikitext', 'subset': "wikitext-2-v1", 'key': 'text'},
        "webtext": {'dataset': 'stas/openwebtext-10k', 'subset': None, 'key': 'text'},
        'glue_rte': {'dataset': 'glue', 'subset': 'rte', 'key': ['sentence1', 'sentence2']},
        # 'glue_mrpc': {'dataset': 'glue', 'subset': 'mrpc', 'key': ['sentence1', 'sentence2']},
        'glue_mrpc': {'dataset': 'glue', 'subset': 'mrpc', 'key': 'sentence1'},
    }