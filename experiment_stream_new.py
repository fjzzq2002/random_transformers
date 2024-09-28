#!/usr/bin/env python
# coding: utf-8

# %%
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, PreTrainedTokenizer, PreTrainedTokenizerBase
from datasets import load_dataset, Dataset
import random
import numpy as np
import wandb
import os
import random
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence
from tqdm.auto import tqdm
from transformers.trainer_callback import TrainerCallback
import wandb

CustomTrainer = Trainer # should be overwritten by task file


arch = 'transformer'
train_steps = 250
seed = 43
n_embd = 16
n_layer = 2
n_head = 4
n_positions = 40
device = 'cuda'
weight_decay = 0.001
learning_rate = 1e-3 if arch=='transformer' else 5e-3
warmup_steps = 500
save_steps = 600
eval_steps = 50
logging_steps = 600
save_total_limit = 3
evaluation_strategy = "steps"
lr_scheduler_type = "cosine"
random_str = ''.join([random.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(8)])
task = 'inductionheadstream_30'
calc_pca = False #True #False
tie_emb = False
weight_frozen = 1
vocab_size = 128
train_size = 40000
eval_size = 4000
dropout = 0.
eval_accumulation_steps = 1

# # %% config from command line
exec(open('configurator.py').read()) # overrides from command line or config file
print(task,flush=True)

# guarantee a deterministic data generation process
random.seed(12345)

task_name = task.split('_')[0]
exec(open('task_'+task_name+'.py').read())

exec(open('configurator.py').read()) # re-override

# %% generate data
taskA_test = taskA_gen()
# assert everything is within vocab size
for d in taskA_test: assert all([0<=x<vocab_size for x in d])
# shuffle
random.shuffle(taskA_test)
assert len(taskA_test) > 0 #and len(taskB_test) > 0
print(f'Task A test: {len(taskA_test)}')
taskA_test_set = set(tuple(x) for x in taskA_test)

# %% generate run description
run_description = task+' w'+str(n_embd)+' f'+str(weight_frozen)+' l'+str(n_layer)+' '+arch
# create folder task if not exists
if not os.path.exists(task):
    os.makedirs(task)
save_path = task+'/'+(random_str+' '+task+' '+' '+run_description).replace('(','_').replace(')','').replace(' ','_').replace(',','_')
print('Run',run_description)
print('Path',save_path)

# %% seed everything
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# %% init wandb
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
print(config)
wandb.init(project='rand_transformer_vwfn', name=run_description, config=config)
wandb.run.log_code(".")

# %% define the model
if arch == 'transformer':
    config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        tie_word_embeddings=tie_emb,
        n_positions=n_positions,
        resid_pdrop = dropout,
        embd_pdrop = dropout,
        attn_pdrop = dropout,
    )
    model = GPT2LMHeadModel(config).to(device)
    if weight_frozen == 101:
        # initialize a sinusoidal position embedding
        n_pos = n_positions
        pos_emb = torch.zeros(n_pos, n_embd)
        position = torch.arange(0, n_pos).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * (-np.log(10000.0) / n_embd))
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        # pos_emb = pos_emb.unsqueeze(0)
        model.transformer.wpe.weight.data = pos_emb
        print(pos_emb)
else:
    # Define the RNN model
    class VanillaRNN(torch.nn.Module):
        def __init__(self, vocab_size, n_embd, n_layer):
            super(VanillaRNN, self).__init__()
            self.encoder = torch.nn.Embedding(vocab_size, n_embd)
            self.rnn = {'rnn':torch.nn.RNN,'lstm':torch.nn.LSTM}[arch](n_embd, n_embd, n_layer, batch_first=True, dropout=dropout)
            self.decoder = torch.nn.Linear(n_embd, vocab_size)

        def forward(self, input_ids, labels=None, hidden=None):
            encoded = self.encoder(input_ids)
            output, hidden = self.rnn(encoded, hidden)
            decoded = self.decoder(output)
            return {'logits':decoded}#, 'hidden':hidden}

    # Instantiate the model
    model = VanillaRNN(vocab_size, n_embd, n_layer).to(device)
model_size = sum(t.numel() for t in model.parameters())
print(f'Arch: {arch}')
print(f"Model size: {model_size/1000**2:.1f}M parameters")


# %% build dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data):
        super().__init__()
        self.input_ids = tokenized_data.clone()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx],
                "labels": self.input_ids[idx]}


'''
Note that if it's a torch.utils.data.IterableDataset with some randomization
and you are training in a distributed fashion, your iterable dataset should
either use a internal attribute generator that is a torch.Generator for the
randomization that must be identical on all processes (and the Trainer will
manually set the seed of this generator at each epoch) or have a set_epoch()
method that internally sets the seed of the RNGs used.

Since we're currently not parallelizing we should be fine for now.
'''
    
class CustomIterDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        super().__init__()

    # def __len__(self):
    #     return train_size
    
    def __iter__(self):
        return iter(self.generate())

    def generate(self):
        while True:
            x = gen_single()
            if tuple(x) in taskA_test_set:
                continue
            yield {"input_ids": x,
                   "labels": x}
# Load your custom tokenized dataset (assuming each example is a list of token IDs)
# Padding is currently not implemented
datasets = {'trainA': None, 'testA': taskA_test}
datasets = {key: CustomDataset(tokenized_data=torch.tensor(val, dtype=torch.long))
            if val is not None and len(val) else CustomIterDataset() for key, val in datasets.items()}

print(datasets.keys())

for p,q in datasets.items():
    print(f'{p}: {q}')

print('Task A test:',len(datasets['testA']))

# %% some training utils

compute_metrics # it should be defined in the task file

from sklearn.decomposition import PCA
class L2NormCallback(TrainerCallback):
    def __init__(self, phase):
        self.phase = phase
        self.pcas = []
        # self.step0 = None

    def on_train_begin(self, args, state, control, **kwargs):
        # self.step0 = state.global_step
        pass

    def on_step_end(self, args, state, control, **kwargs):
        # if state.global_step == self.step0:
        #     return
        model = kwargs['model']
        l2_norm = sum([p.norm()**2 for p in model.parameters()]).sqrt()
        wandb.log({'l2_norm_all': l2_norm.item()})
        wandb.log({self.phase+'_l2_norm_all': l2_norm.item()})
        l2_norm_noln = sum([p.norm()**2 for n,p in model.named_parameters() if '.ln' not in n]).sqrt()
        wandb.log({'l2_norm': l2_norm_noln.item()})
        wandb.log({self.phase+'_l2_norm': l2_norm_noln.item()})
        if calc_pca:
            # calculate PCA of input embeddings
            embds = model.transformer.wte.weight
            pca = PCA(n_components=20)
            embd_PCA = pca.fit_transform(embds.detach().cpu().numpy())
            wandb.log({'PCA_explained_variance_ratio': pca.explained_variance_ratio_.tolist()})
            self.pcas.append(embd_PCA)
    
    def save_pca(self):
        import pickle
        pca_log = np.array(self.pcas)
        with open("./"+save_path+'/pca_'+self.phase+'.pickle', 'wb') as f:
            pickle.dump(pca_log, f)

# %% training arguments
training_args_dict = dict(
    output_dir="./"+save_path,
    overwrite_output_dir=True,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    save_steps=save_steps,
    evaluation_strategy=evaluation_strategy,
    eval_steps=eval_steps,
    logging_steps=logging_steps,
    save_total_limit=save_total_limit,
    weight_decay=weight_decay,
    warmup_steps=warmup_steps,
    lr_scheduler_type=lr_scheduler_type,
    learning_rate=learning_rate,
    run_name=run_description,
)

# %% train
wandb.log({'phase': 'train'})
num_frozen = 0
num_trainable = 0

for n,p in model.named_parameters():
    if weight_frozen==2 and not any(x in n for x in ['lm_head', 'decoder']):
        print(n,p.shape,'frozen')
        p.requires_grad=False
    elif weight_frozen==3 and not any(x in n for x in ['wte','wpe','encoder']):
        print(n,p.shape,'frozen')
        p.requires_grad=False
    elif weight_frozen in [100,101] and not any(x in n for x in ['wte','lm_head','encoder','decoder']):
        print(n,p.shape,'frozen')
        p.requires_grad=False
    elif weight_frozen==1 and not any(x in n for x in ['wte','lm_head','wpe','encoder','decoder']):
        print(n,p.shape,'frozen')
        p.requires_grad=False
    else:
        print('trainable:',n,p.shape)
    if p.requires_grad:
        num_trainable += p.numel()
    else:
        num_frozen += p.numel()

print(f'# parameters: {num_trainable} trainable, {num_frozen} frozen')
wandb.log({'num_trainable': num_trainable})
wandb.log({'num_frozen': num_frozen})

training_args = TrainingArguments(
    max_steps=train_steps*train_size//per_device_train_batch_size,
    **training_args_dict
)
    
train_logger = L2NormCallback('train')
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=(
        datasets['trainA']
    ),
    eval_dataset={'train_'+a:b for a,b in datasets.items() if a.startswith('test')},
    compute_metrics=compute_metrics,
    callbacks=[train_logger],
)

trainer.train()