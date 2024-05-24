import torch
from torch import nn
from torch.nn.parameter import Parameter
from models.layers import *
from torch.distributions.gamma import Gamma
from util.utils import get_token_embedding
from data.load_data import sample_noise_Gaussian, sample_noise_Chi
from util.globals import *
from baseline.ks_dist import ddKS
from transformers import BartForConditionalGeneration, AutoTokenizer

class NoisyModel(nn.Module):
    def __init__(self, base_model, args):
        super().__init__()
        self.base_model = base_model
        self.noise_mechanism = args.noise_mechanism
        if args.noise_mechanism == "Gaussian":
            self.noise_stddev = args.noise_std
        elif args.noise_mechanism == "ChiDP":
            self.eta = args.test_eta
        self.args = args

    def forward(self, input_ids, attention_mask=None, decoder_start_token=None):
        # Get the embeddings
        embeddings = get_token_embedding(input_ids, self.base_model, self.args)
        if self.eta > 0:
            if self.noise_mechanism == "Gaussian":
                noise = sample_noise_Gaussian(init_emb.shape, self.noise_stddev, self.args.device)
            elif self.noise_mechanism == "ChiDP":
                noise = sample_noise_Chi(embeddings.shape, self.eta, self.args.device)
        else:
            noise = 0
        # Add the noise to the embeddings
        noisy_embeddings = embeddings + noise

        # Feed the noisy embeddings into the rest of the model
        with torch.no_grad():
            if "t5" in self.args.base_model:
                decoder_start_token = decoder_start_token.repeat(noisy_embeddings.shape[0], 1)
                outputs = self.base_model(inputs_embeds=noisy_embeddings, 
                                        attention_mask=attention_mask,
                                        decoder_input_ids=decoder_start_token,
                                        output_hidden_states=True)
            else:
                outputs = self.base_model(inputs_embeds=noisy_embeddings, 
                                        attention_mask=attention_mask,
                                        output_hidden_states=True)

        return outputs
    
class ScaleModel(nn.Module):
    def __init__(self, base_model, args):
        super().__init__()
        self.base_model = base_model
        self.noise_mechanism = args.noise_mechanism
        if args.noise_mechanism == "Gaussian":
            self.noise_stddev = args.noise_std
        elif args.noise_mechanism == "ChiDP":
            self.eta = args.test_eta
        self.args = args
        # sample Chi distribution
        if self.eta > 0:
            d_shape = (1000, 8)
            if args.noise_mechanism == "Gaussian":
                self.ref_noises = sample_noise_Gaussian(d_shape, self.noise_stddev, args.device)
            elif args.noise_mechanism == "ChiDP":
                self.ref_noises = sample_noise_Chi(d_shape, self.eta, args.device)
            self.ks_cal = ddKS()

    def forward(self, input_ids, attention_mask=None, decoder_start_token=None):
        # Get the embeddings
        embeddings = get_token_embedding(input_ids, self.base_model, self.args)
        if self.eta > 0:
            if self.noise_mechanism == "Gaussian":
                noise = sample_noise_Gaussian(init_emb.shape, self.noise_stddev, self.args.device)
            elif self.noise_mechanism == "ChiDP":
                noise = sample_noise_Chi(embeddings.shape, self.eta, self.args.device)
        else:
            noise = 0
        # Add the noise to the embeddings
        noisy_embeddings = embeddings + noise
        # compute the ks distance with reference distributions
        if self.eta > 0:
            ks_dists = []
            for noisy_emb in noisy_embeddings:
                reshaped_noisy_emb = noisy_emb.reshape(-1, 8)
                this_ks = self.ks_cal(reshaped_noisy_emb[:500], self.ref_noises)
                ks_dists.append(this_ks)
            ks_dists = torch.tensor(ks_dists) 
            ks_dists = ks_dists.to(noisy_embeddings.device)
            noisy_embeddings = torch.einsum('i,ijk->ijk', ks_dists, noisy_embeddings)
        # Feed the noisy embeddings into the rest of the model
        with torch.no_grad():
            if "t5" in self.args.base_model:
                decoder_start_token = decoder_start_token.repeat(noisy_embeddings.shape[0], 1)
                outputs = self.base_model(inputs_embeds=noisy_embeddings, 
                                        attention_mask=attention_mask,
                                        decoder_input_ids=decoder_start_token,
                                        output_hidden_states=True)
            else:
                outputs = self.base_model(inputs_embeds=noisy_embeddings, 
                                        attention_mask=attention_mask,
                                        output_hidden_states=True)

        return outputs

class PARTModel(nn.Module):
    def __init__(self, base_model, n_labels, args):
        super().__init__()
        self.base_model = base_model
        self.args = args
        word_dim = emb_size_dict[args.base_model]
        self.prompt_emb = nn.Parameter(torch.randn(args.num_virtual_tokens, word_dim))
        self.linear_sig = nn.Sequential(
                nn.Linear(word_dim, args.rec_vocab_size),
                nn.Softmax(),
            )
        self.classifier = nn.Sequential(
                nn.Linear(word_dim, n_labels),
                nn.Softmax(),
            )
        self.classifier_loss = nn.CrossEntropyLoss()

    def get_hid_states(self, outputs):
        if self.args.base_model =="stevhliu/my_awesome_model":
            hid_states = outputs.hidden_states[-1]
        elif self.args.base_model in ("bert-base-uncased", "bert-large-uncased"):
            hid_states = outputs.hidden_states[-1]
        elif "gpt2" in self.args.base_model:
            hid_states = outputs.last_hidden_state
        elif "t5" in self.args.base_model:
            hid_states = outputs.last_hidden_state
        elif any(model in self.args.base_model for model in ['opt', 'llama']):
            hid_states = outputs.hidden_states[-1]
        return hid_states

    def get_cls_embedding(self, hidden_states, attention_mask):
        if any(model in self.args.base_model for model in ['opt', 'llama', "gpt2"]):
            sum_attention_mask = attention_mask.sum(dim=1)
            last_pad = torch.where(sum_attention_mask > 0, sum_attention_mask - 1, torch.zeros_like(sum_attention_mask))
            last_pad = last_pad.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, hidden_states.shape[-1])
            cls_embs = torch.gather(hidden_states, index=last_pad, dim=1)
            cls_embs = cls_embs.squeeze()
        elif any(model in self.args.base_model for model in ['stevhliu/my_awesome_model', 'bert', "t5"]):
            hidden_states = hidden_states[:,(self.args.num_virtual_tokens+self.args.n_plain_tok):,:]
            cls_embs = hidden_states[:,0,:]
        return cls_embs

    def forward(self, input_ids, plain_tokens, attention_mask, labels=None, decoder_start_token=None):
        # Get the embeddings
        embeddings = get_token_embedding(input_ids, self.base_model, self.args)
        embeddings = embeddings[:, :-self.args.num_virtual_tokens, :]
        # Concate with virtual prompt embedding
        prompt_emb = self.prompt_emb.unsqueeze(0).repeat(input_ids.shape[0], 1, 1)
        embeddings = torch.cat([prompt_emb, embeddings], dim = 1)
        virtual_tokens_mask = torch.ones(input_ids.shape[0], self.args.num_virtual_tokens)
        virtual_tokens_mask = virtual_tokens_mask.to(self.args.device)
        attention_mask = torch.cat((virtual_tokens_mask, attention_mask), dim=1)
        attention_mask = attention_mask[:, :-self.args.num_virtual_tokens]

        # Feed the noisy embeddings into the rest of the model
        with torch.no_grad():
            if "t5" in self.args.base_model:
                decoder_start_token = decoder_start_token.repeat(embeddings.shape[0], 1)
                outputs = self.base_model(inputs_embeds=embeddings, 
                                        attention_mask=attention_mask,
                                        decoder_input_ids=decoder_start_token,
                                        output_hidden_states=True)
            else:
                outputs = self.base_model(inputs_embeds=embeddings, 
                                        attention_mask=attention_mask,
                                        output_hidden_states=True)
        hid_states = self.get_hid_states(outputs)
        # classification layer
        cls_emb = self.get_cls_embedding(hid_states, attention_mask)
        class_prob = self.classifier(cls_emb)
        if labels is not None:
            # plain token reconstruction layer
            plain_tok_states = hid_states[:,self.args.num_virtual_tokens:(self.args.num_virtual_tokens+self.args.n_plain_tok),:]
            plain_tok_probs = self.linear_sig(plain_tok_states)
            plain_tokens = plain_tokens.unsqueeze(-1)
            plain_tok_probs = (torch.gather(plain_tok_probs, index=plain_tokens, dim=-1)).squeeze()
            plain_tok_probs = torch.log(plain_tok_probs + 1e-8)
            # compute task specific loss
            loss = self.classifier_loss(class_prob, labels)
            loss -= torch.sum(plain_tok_probs)
            return loss
        else:
            return class_prob


class clsModel(nn.Module):
    def __init__(self, input_dim, n_class):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, n_class),
            nn.LeakyReLU(),
        )
        torch.manual_seed(1234)
        self.linear_relu_stack.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class regModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(),
            nn.Linear(input_dim, 1),
            nn.LeakyReLU(),
        )
        torch.manual_seed(1234)
        self.linear_relu_stack.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output

class EnhancedClsModel(nn.Module):
    def __init__(self, input_dim, n_class):
        super().__init__()

        hidden_dim = input_dim * 8 # 增加隐藏层维度
        dropout_rate=0.5

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.BatchNorm1d(hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_dim // 8, n_class) # 输出层
        )

        # 初始化权重
        self.model.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.01)

    def forward(self, x):
        logits = self.model(x)
        return logits

class linearModel(nn.Module):
    def __init__(self, input_dim, n_class):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_class-1)
        # self.linear = nn.Sequential(
        #     nn.Linear(input_dim, 2*input_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(2*input_dim, input_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(input_dim, n_class-1),
        # )
        torch.manual_seed(1234)
        self.linear.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        logits = self.linear(x)
        outputs = torch.sigmoid(logits)
        return outputs

class clsModel1(nn.Module):
    def __init__(self, input_dim, n_class):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.LeakyReLU(),
            nn.Linear(input_dim//2, input_dim//4),
            nn.LeakyReLU(),
            nn.Linear(input_dim//4, n_class),
        )
        torch.manual_seed(1234)
        self.linear_relu_stack.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class clsModel2(nn.Module):
    def __init__(self, input_dim, n_class):
        super().__init__()
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, n_class)
        )
        torch.manual_seed(1234)
        self.linear_tanh_stack.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        logits = self.linear_tanh_stack(x)
        return logits

class mlpModel(nn.Module):
    def __init__(self, input_dim):
        super(mlpModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Assuming binary classification
        )

    def forward(self, x):
        return self.layers(x)
class AttributeInferenceMLP(nn.Module):
    def __init__(self, input_dim, output_dim=2):  # Assuming binary classification
        super(AttributeInferenceMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 768)  # First fully connected layer
        self.relu = nn.ReLU()  # ReLU activation
        self.fc2 = nn.Linear(768, output_dim)  # Second fully connected layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class mlpModel1(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_class):
        super(mlpModel1, self).__init__()
        
        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LeakyReLU())
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, n_class))
        
        self.model = nn.Sequential(*layers)
        self.model.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.model(x)

class mlpModel2(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_class, dropout_rate=0.5):
        super(mlpModel2, self).__init__()
        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))  # Batch Normalization
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_rate))  # Dropout
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, n_class))
        self.model = nn.Sequential(*layers)
        self.model.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.model(x)

class serverDenoiseModel(nn.Module):
    def __init__(self, d_model, args, decoder=False): #change some attributes
        super(serverDenoiseModel, self).__init__()
        self.args = args
        self.decoder = decoder
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.input_ids = tokenizer("denoise: ", return_tensors='pt').input_ids
        self.input_ids = self.input_ids.to(args.device)
        self.base_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
        self.emb_map = nn.Linear(d_model, 1024)
    
    def forward(self, noisy_embedding, attention_mask=None, labels=None):
        #print('here is the forward')
        mapped_embs = self.emb_map(noisy_embedding)
        input_ids = self.input_ids.repeat(mapped_embs.shape[0], 1)
        if labels is not None:
            ignore_index=-100
            labels = labels.masked_fill(attention_mask == 0, ignore_index)
        outputs = self.base_model(input_ids=input_ids, decoder_attention_mask=attention_mask, 
                                  decoder_inputs_embeds=mapped_embs, labels=labels)
        return outputs

class denoiseModel(nn.Module):
    def __init__(self, d_model, d_out, args, decoder=False): #change some attributes
        super(denoiseModel, self).__init__()
        self.n_emb_block = args.n_emb_block
        self.n_noise_block = args.n_noise_block
        self.d_model = d_model
        self.d_out = d_out
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers
        self.d_ff = args.d_ff
        self.att_pool = args.att_pool
        self.comb = args.comb
        if self.att_pool:
            self.emb_transformer = nn.ModuleList([TransformerAP(d_model, d_out, args.num_heads, args.num_layers, args.dim_head, args.d_ff, args.dropout, decoder) 
                                         for _ in range(args.n_emb_block)])
            self.noise_transformer = nn.ModuleList([TransformerAP(d_model, d_out, args.num_heads, args.num_layers, args.dim_head, args.d_ff, args.dropout, decoder) 
                                            for _ in range(args.n_noise_block)])
        else:
            self.emb_transformer = nn.ModuleList([Transformer(d_model, d_out, args.num_heads, args.num_layers, args.dim_head, args.d_ff, args.dropout, decoder) 
                                            for _ in range(args.n_emb_block)])
            self.noise_transformer = nn.ModuleList([Transformer(d_model, d_out, args.num_heads, args.num_layers, args.dim_head, args.d_ff, args.dropout, decoder) 
                                            for _ in range(args.n_noise_block)])
        if self.comb == "MLP" or self.comb == "MLP_v2" or self.comb == "MLP_v3":
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(d_out, d_out),
                nn.LeakyReLU(),
                nn.Linear(d_out, d_out),
                nn.LeakyReLU(),
            )
        elif self.comb == "attention" or self.comb == "att_w_output" or self.comb == "att_w_output_v2" or self.comb == "att_w_output_v3":
            self.attention_pool = AttentionPooling(d_out, d_out, args.dropout)
        if self.comb == "att_w_output_v2" or self.comb == "MLP_v2":
            self.linear_tanh_stack = nn.Sequential(
                nn.Linear(2*d_out, d_out),
                nn.Tanh(),
                nn.Linear(d_out, d_out),
            )
        if self.comb == "att_w_output_v3" or self.comb == "MLP_v3":
            self.linear = nn.Linear(2*d_out, d_out)
    
    def forward(self, init_embedding, noise, output, attention_mask=None):
        #print('here is the forward')
        for layer in self.emb_transformer:
            init_embedding = layer(init_embedding, mask=attention_mask)
        for layer in self.noise_transformer:
            noise = layer(noise, mask=attention_mask)
        if self.comb == "MLP":
            diff = init_embedding + noise
            diff = self.linear_relu_stack(diff)
            output = output - diff
        elif self.comb == "MLP_v2":
            diff = init_embedding + noise
            diff = self.linear_relu_stack(diff)
            output = torch.cat([diff, output], dim=-1)
            output = self.linear_tanh_stack(output)
        elif self.comb == "MLP_v3":
            diff = init_embedding + noise
            diff = self.linear_relu_stack(diff)
            output = torch.cat([diff, output], dim=-1)
            output = self.linear(output)
        elif self.comb == "attention":
            diff = torch.stack([init_embedding, noise], dim=1)
            diff = self.attention_pool(diff)
            output = output - diff
        elif self.comb == "att_w_output":
            output = torch.stack([init_embedding, noise, output], dim=1)
            output = self.attention_pool(output)
        elif self.comb == "att_w_output_v2":
            diff = torch.stack([init_embedding, noise], dim=1)
            diff = self.attention_pool(diff)
            output = torch.cat([diff, output], dim=-1)
            output = self.linear_tanh_stack(output)
        elif self.comb == "att_w_output_v3":
            diff = torch.stack([init_embedding, noise], dim=1)
            diff = self.attention_pool(diff)
            output = torch.cat([diff, output], dim=-1)
            output = self.linear(output)
        return output

    def get_parameter(self):
        return self.n_emb_block, self.n_noise_block, self.d_model, self.d_out, self.num_heads, self.num_layers, self.d_ff
    
class denoiseModelv2(nn.Module):
    def __init__(self, d_model, d_out, args, decoder=False): #change some attributes
        super(denoiseModelv2, self).__init__()
        self.n_emb_block = args.n_emb_block
        self.n_noise_block = args.n_noise_block
        self.d_model = d_model
        self.d_out = d_out
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers
        self.d_ff = args.d_ff
        self.att_pool = args.att_pool
        self.comb = args.comb
        self.linear_input_first = nn.Linear(2*d_out, d_out)
        self.activation = nn.Tanh()
        self.linear_input_sec = nn.Linear(d_out, d_out)
        if self.att_pool:
            self.transformer = nn.ModuleList([TransformerAP(d_model, d_out, args.num_heads, args.num_layers, args.dim_head, args.d_ff, args.dropout, decoder) 
                                         for _ in range(args.n_emb_block)])
        else:
            self.transformer = nn.ModuleList([Transformer(d_model, d_out, args.num_heads, args.num_layers, args.dim_head, args.d_ff, args.dropout, decoder) 
                                            for _ in range(args.n_emb_block)])
        if self.comb == "MLP":
            self.linear_tanh_stack = nn.Sequential(
                nn.Linear(2*d_out, d_out),
                nn.Tanh(),
                nn.Linear(d_out, d_out),
            )
        if self.comb == "MLP_v2":
            self.linear = nn.Linear(2*d_out, d_out)
        elif self.comb == "attention":
            self.attention_pool = AttentionPooling(d_out, d_out, args.dropout)
    
    def forward(self, init_embedding, noise, output, attention_mask=None):
        input = torch.cat([init_embedding, noise], dim=-1)
        input = self.linear_input_first(input)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand_as(input)
            input = input.masked_fill(mask == 0, 0)
        input = self.activation(input)
        input = self.linear_input_sec(input)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand_as(input)
            input = input.masked_fill(mask == 0, 0)
        for layer in self.transformer:
            input = layer(input, mask=attention_mask)
        if self.comb == "MLP":
            output = torch.cat([input, output], dim=-1)
            output = self.linear_tanh_stack(output)
        if self.comb == "MLP_v2":
            output = torch.cat([input, output], dim=-1)
            output = self.linear(output)
        elif self.comb == "attention":
            output = torch.stack([input, output], dim=1)
            output = self.attention_pool(output)
        return output

    def get_parameter(self):
        return self.n_emb_block, self.n_noise_block, self.d_model, self.d_out, self.num_heads, self.num_layers, self.d_ff

class denoiseModelCrsAtt(nn.Module):
    def __init__(self, d_model, d_out, args): #change some attributes
        super(denoiseModelCrsAtt, self).__init__()
        self.n_emb_block = args.n_emb_block
        self.n_noise_block = args.n_noise_block
        self.d_model = d_model
        self.d_out = d_out
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers
        self.d_ff = args.d_ff
        self.att_pool = args.att_pool
        self.comb = args.comb
        self.linear_input_first = nn.Linear(2*d_out, d_out)
        self.activation = nn.Tanh()
        self.linear_input_sec = nn.Linear(d_out, d_out)
        self.transformer = nn.ModuleList([CrsTransformer(d_model, d_out, args.num_heads, args.num_layers, args.dim_head, args.d_ff, args.dropout) 
                                         for _ in range(args.n_emb_block)])

        if self.comb == "att_att_v1":
            self.attn_pool1 = AttentionPooling(d_out, d_out, args.dropout)
            self.attn_pool2 = AttentionPooling(d_out, d_out, args.dropout)
            self.attn_pool_diff = AttentionPooling(d_out, d_out, args.dropout)
            self.linear_tanh_stack = nn.Sequential(
                    nn.Linear(2*d_out, d_out),
                    nn.Tanh(),
                    nn.Linear(d_out, d_out),
                )
        
        elif self.comb == "select_att_v1":
            self.attn_pool_diff = AttentionPooling(d_out, d_out, args.dropout)
            self.linear_tanh_stack = nn.Sequential(
                    nn.Linear(2*d_out, d_out),
                    nn.Tanh(),
                    nn.Linear(d_out, d_out),
                )
        
    def forward(self, init_embedding, noise, output, attention_mask=None):
        for layer in self.transformer:
            init_embedding, noise = layer(init_embedding, noise, mask=attention_mask)

        if self.comb == "att_att_v1":
            init_embedding = self.attn_pool1(init_embedding, attention_mask)
            noise = self.attn_pool2(noise, attention_mask)
            diff = torch.stack([init_embedding, noise], dim=1)
            diff = self.attn_pool_diff(diff)
            output = torch.cat([diff, output], dim=-1)
            output = self.linear_tanh_stack(output)

        if self.comb == "select_att_v1":
            init_embedding = init_embedding[:, 0, :]
            noise = noise[:, 0, :]
            diff = torch.stack([init_embedding, noise], dim=1)
            diff = self.attn_pool_diff(diff)
            output = torch.cat([diff, output], dim=-1)
            output = self.linear_tanh_stack(output)
        return output

class denoiseModelCrsAttv2(nn.Module):
    def __init__(self, d_model, d_out, args): #change some attributes
        super(denoiseModelCrsAttv2, self).__init__()
        self.n_emb_block = args.n_emb_block
        self.n_noise_block = args.n_noise_block
        self.d_model = d_model
        self.d_out = d_out
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers
        self.d_ff = args.d_ff
        self.att_pool = args.att_pool
        self.comb = args.comb
        self.linear_input_first = nn.Linear(2*d_out, d_out)
        self.activation = nn.Tanh()
        self.linear_input_sec = nn.Linear(d_out, d_out)
        self.transformer = nn.ModuleList([CrsTransformer(d_model, d_out, args.num_heads, args.num_layers, args.dim_head, args.d_ff, args.dropout) 
                                         for _ in range(args.n_emb_block)])
        if self.comb == "att_att_v1":
            self.attn_pool1 = AttentionPooling(d_out, d_out, args.dropout)
            self.attn_pool2 = AttentionPooling(d_out, d_out, args.dropout)
            self.attn_pool_diff = AttentionPooling(d_out, d_out, args.dropout)
        
        elif self.comb == "select_att_v1":
            self.attn_pool_diff = AttentionPooling(d_out, d_out, args.dropout)

    def forward(self, init_embedding, noise, output, attention_mask=None):
        init_embedding = torch.cat([output.unsqueeze(dim=1), init_embedding], dim = 1)
        noise = torch.cat([output.unsqueeze(dim=1), noise], dim = 1)
        if attention_mask is not None:
            batch_size = output.shape[0]
            ones_ts = torch.ones(batch_size, 1, device = attention_mask.device)
            attention_mask = torch.cat([ones_ts, attention_mask], dim=1)
        for layer in self.transformer:
            init_embedding, noise = layer(init_embedding, noise, mask=attention_mask)

        if self.comb == "att_att_v1":
            init_embedding = self.attn_pool1(init_embedding, attention_mask)
            noise = self.attn_pool2(noise, attention_mask)
            output = torch.stack([init_embedding, noise], dim=1)
            output = self.attn_pool_diff(output)

        if self.comb == "select_att_v1":
            init_embedding = init_embedding[:, 0, :]
            noise = noise[:, 0, :]
            output = torch.stack([init_embedding, noise], dim=1)
            output = self.attn_pool_diff(output)

        return output

class denoiseModelv3(nn.Module):
    def __init__(self, d_model, d_out, args, decoder=False): #change some attributes
        super(denoiseModelv3, self).__init__()
        self.n_emb_block = args.n_emb_block
        self.n_noise_block = args.n_noise_block
        self.d_model = d_model
        self.d_out = d_out
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers
        self.d_ff = args.d_ff
        self.att_pool = args.att_pool
        self.comb = args.comb
        self.linear_input_first = nn.Linear(2*d_out, d_out)
        self.activation = nn.Tanh()
        self.linear_input_sec = nn.Linear(d_out, d_out)
        self.transformer = Transformer(d_model, d_out, args.num_heads, args.dim_head, args.num_layers, args.d_ff, args.dropout, decoder)

    def forward(self, init_embedding, noise, output, attention_mask=None):
        output = torch.cat([output.unsqueeze(dim=1), init_embedding, noise], dim = 1)
        if attention_mask is not None:
            batch_size = output.shape[0]
            ones_ts = torch.ones(batch_size, 1, device = attention_mask.device)
            attention_mask = torch.cat([ones_ts, attention_mask, attention_mask], dim=1)
        output = self.transformer(output, mask=attention_mask)
        return output

class denoiseModelv4(nn.Module):
    def __init__(self, d_model, d_out, args, decoder=False): #change some attributes
        super(denoiseModel, self).__init__()
        self.n_emb_block = args.n_emb_block
        self.n_noise_block = args.n_noise_block
        self.d_model = d_model
        self.d_out = d_out
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers
        self.d_ff = args.d_ff
        self.att_pool = args.att_pool
        self.comb = args.comb
        self.linear_tanh_stack = nn.Sequential(
                nn.Linear(2*d_out, d_out),
                nn.Tanh(),
                nn.Linear(d_out, d_out),
            )
        self.emb_transformer = Transformer(d_model, d_out, args.num_heads, args.dim_head, args.num_layers, args.d_ff, args.dropout, decoder)
        self.noise_transformer = Transformer(d_model, d_out, args.num_heads, args.dim_head, args.num_layers, args.d_ff, args.dropout, decoder)
    
    def forward(self, init_embedding, noise, output, attention_mask=None):
        if attention_mask is not None:
            batch_size = output.shape[0]
            ones_ts = torch.ones(batch_size, 1, device = attention_mask.device)
            attention_mask = torch.cat([ones_ts, attention_mask], dim=1)
        output_emb = torch.cat([output.unsqueeze(dim=1), init_embedding], dim = 1)
        output_emb = self.transformer(output_emb, mask=attention_mask)
        output_noise = torch.cat([output.unsqueeze(dim=1), noise], dim = 1)
        output_noise = self.transformer(output_noise, mask=attention_mask)
        output = torch.cat([output_emb, output_noise], dim=-1)
        output = self.linear_tanh_stack(output)
        return output

    def get_parameter(self):
        return self.n_emb_block, self.n_noise_block, self.d_model, self.d_out, self.num_heads, self.num_layers, self.d_ff

if __name__ == "__main__":
    from util.parameters import get_args
    import numpy as np
    from transformers import (AutoModelForSequenceClassification,
                              GPT2Model,
                              AutoModelForCausalLM,
                              BertModel)
    args = get_args()
    # args.dim_head = 240
    # model = denoiseModelv3(d_model=4096, d_out=4096, args=args)
    # model = denoiseModel(d_model=1600, d_out=1600, args=args)
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    # # model = GPT2Model.from_pretrained('gpt2-xl')
    # params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # param_dict = {name: p for name, p in model.named_parameters() if p.requires_grad}
    
    model = BertModel.from_pretrained("bert-base-uncased")
    input_ids = torch.tensor([1,2,3,4])
    init_emb = model.embeddings.word_embeddings(input_ids)
    a = 1