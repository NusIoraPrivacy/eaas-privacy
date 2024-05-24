from models.model import *
from data.load_data import *
from denoise import preprocess_data, load_train, train_denoise
from util.parameters import get_args
import os
from util.utils import pt_to_hdf5, get_pretrained_model, get_finetuned_model
from huggingface_hub import login
login(token="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")

args = get_args()
if args.use_ft_base:
    tokenizer, base_model = get_finetuned_model(args)
else:
    tokenizer, base_model = get_pretrained_model(args)

# if not os.path.exists(args.denoise_model_dir):
#     os.makedirs(args.denoise_model_dir, exist_ok=True)

if args.train_sync:
    denoise_mod = train_denoise(base_model= base_model, args=args, tokenizer=tokenizer)
###Load the training dataset 
else:
    if args.gen_data:
        input_len = preprocess_data(base_model= base_model, args=args, tokenizer=tokenizer)
        print("Finish saving all embeddings and noises")
    base_mod_name = args.base_model.split('/')[-1]
    if args.noise_mechanism == "Gaussian":
        if args.denoise_data == "mix":
            denoise_data_name = ''.join([i[0] for i in args.denoise_data_subsets])
            denoise_data_dir = os.path.join(args.denoise_data_dir, f"{args.denoise_data}_{denoise_data_name}_noise_{base_mod_name}_{args.noise_mechanism}_{args.noise_std}_{args.clip}")
        else:
            denoise_data_dir = os.path.join(args.denoise_data_dir, f"{args.denoise_data}_noise_{base_mod_name}_{args.noise_mechanism}_{args.noise_std}_{args.clip}")
    elif args.noise_mechanism == "ChiDP":
        if args.denoise_data == "mix":
            denoise_data_name = args.mixed_data_config_list.split(",")
            denoise_data_name = [i.strip()[0] for i in denoise_data_name]
            denoise_data_name = ''.join(denoise_data_name)
            denoise_data_dir = os.path.join(args.denoise_data_dir, f"{args.denoise_data}_{denoise_data_name}_noise_{base_mod_name}_{args.noise_mechanism}_{args.train_eta}_{args.clip}")
        else:
            denoise_data_dir = os.path.join(args.denoise_data_dir, f"{args.denoise_data}_noise_{base_mod_name}_{args.noise_mechanism}_{args.train_eta}_{args.clip}")
    # train_data_path
    train_h5_path = os.path.join(denoise_data_dir, "train_data.hdf5")

    if not os.path.exists(train_h5_path):
        pt_to_hdf5(args)
    # start training denoise model
    load_train(train_h5_path, base_model, args=args)

print("Finish training denoise model")