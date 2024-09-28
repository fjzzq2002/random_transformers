#!/usr/bin/env python
# coding: utf-8


# In[151]:


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



arch = 'transformer'
train_steps = 250
seed = 43
n_embd = 256
n_layer = 3
n_head = 4
n_positions = 40
device = 'cuda' #...
weight_decay = 0.001
learning_rate = 1e-3 if arch=='transformer' else 5e-3
warmup_steps = 500
save_steps = 200
eval_steps = 200
logging_steps = 200
save_total_limit = 3
evaluation_strategy = "steps"
lr_scheduler_type = "cosine"
random_str = ''.join([random.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(8)])
calc_pca = False #True #False
tie_emb = False
weight_frozen = 1
vocab_size = 512
train_size = 40000
eval_size = 4000
dropout = 0.
eval_accumulation_steps = 1
n_embd_target = 512
n_layer_target = 3
n_head_target = 2
per_device_train_batch_size = 1000
per_device_eval_batch_size = 1000
task = None

# # %% config from command line
exec(open('configurator.py').read()) # overrides from command line or config file
task = f'imitationtest_{n_embd_target}_{n_layer_target}_{n_head_target}'
print(task,flush=True)

# guarantee a deterministic data generation process
random.seed(12345)
torch.manual_seed(12345)
np.random.seed(12345)
config = GPT2Config(
    vocab_size=vocab_size,
    n_embd=n_embd_target,
    n_layer=n_layer_target,
    n_head=n_head_target,
    tie_word_embeddings=False,
    n_positions=n_positions,
    resid_pdrop = 0.,
    embd_pdrop = 0.,
    attn_pdrop = 0.,
)
model_imitation = GPT2LMHeadModel(config).to('cpu')
model_imitation.eval()
for u in model_imitation.transformer.h:
    u.mlp.c_proj.weight.data*=20
    u.mlp.c_proj.bias.data*=20
    u.attn.c_attn.weight.data*=10
model_imitation.lm_head.weight.data*=100/n_embd_target**0.5


# In[155]:
def gen_single():
    # generate a random seq of [n_positions] numbers
    l = [random.randint(0,vocab_size-1) for _ in range(n_positions)]
    o = model_imitation.forward(torch.tensor(l,dtype=int,device=model_imitation.device))
    from scipy.special import logsumexp
    logits = o['logits'][-1].detach().cpu().numpy()
    # normalize
    logits = logits - logsumexp(logits)
    return (l, logits)

# In[156]:


def gen_batch(n):
    l = [[random.randint(0,vocab_size-1) for _ in range(n_positions)] for _ in range(n)]
    o = model_imitation.forward(torch.tensor(l,dtype=int,device=model_imitation.device))
    from scipy.special import logsumexp
    logits = o['logits'][:,-1].detach().cpu().numpy()
    # normalize
    logits = logits - logsumexp(logits,axis=1)[:,None]
    return (l, logits)

bt = gen_batch(10000)
def kldivergence(a_logit, b_logit):
    # normalize logits
    a_logit = a_logit - np.log(np.sum(np.exp(a_logit)))
    b_logit = b_logit - np.log(np.sum(np.exp(b_logit)))
    a_prob = np.exp(a_logit)
    return np.sum(a_prob * (a_logit - b_logit))

# In[162]:
kd=[]
for u in range(10000):
    kd.append(kldivergence(random.choice(bt[1]),random.choice(bt[1])))
import matplotlib.pyplot as plt
#print(n_embd_target,np.median(kd),np.std(kd),np.mean(kd))
print(f'{n_embd_target=}',f'{np.median(kd)=:.2f}',f'{np.std(kd)=:.2f}',f'{np.mean(kd)=:.2f}')

def taskA_gen():
    tasks = []
    for i in range(eval_size):
        tasks.append(gen_single())
    return tasks

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        #print(inputs.keys())
        expected = inputs.pop("expected_logits")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        logits = logits[:,-1]
        # kl divergence
        loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(logits, dim=-1), expected, reduction='batchmean', log_target=True)
        return (loss, outputs) if return_outputs else loss

# %% generate data
taskA_test = taskA_gen()
# assert everything is within vocab size
for d,l in taskA_test: assert all([0<=x<vocab_size for x in d])
# shuffle
random.shuffle(taskA_test)
assert len(taskA_test) > 0
print(f'Task A test: {len(taskA_test)}')
taskA_test_set = set(tuple(x) for x,l in taskA_test)

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
wandb.init(project='rand_transformer_vwfn_imitation', name=run_description, config=config)
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
    model = GPT2LMHeadModel(config)
    model = torch.nn.DataParallel(model)
    model = model.to(device)
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

model.forward_=model.forward
def forward(self, *args, **kwargs):
    expected_logits = kwargs.pop('expected_logits', None)
    output = self.forward_(*args, **kwargs)
    return output
import types
model.forward = types.MethodType(forward, model)

model_size = sum(t.numel() for t in model.parameters())
print(f'Arch: {arch}')
print(f"Model size: {model_size/1000**2:.1f}M parameters")


for a,p in model.named_parameters():
    print(a,f'{p.shape=}',f'{p.mean().item()=}',f'{p.std().item()=}')


# %% build dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.input_ids = torch.tensor([x[0] for x in data], dtype=torch.long)
        self.expected_logits = torch.tensor([x[1] for x in data], dtype=torch.float32)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx],
                "expected_logits": self.expected_logits[idx]}


# In[28]:

class CustomIterDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        super().__init__()
    
    def __iter__(self):
        return iter(self.generate())

    def generate(self):
        while True:
            x = gen_batch(10000)
            for p,q in zip(x[0],x[1]):
                if tuple(p) in taskA_test_set:
                    continue
                yield {"input_ids": p,
                       "expected_logits": q}


# In[29]:

datasets = {'trainA': None, 'testA': taskA_test}
datasets = {key: CustomDataset(val)
            if val is not None and len(val) else CustomIterDataset() for key, val in datasets.items()}


# In[30]:

print(datasets.keys())

for p,q in datasets.items():
    print(f'{p}: {q}')

print('Task A test:',len(datasets['testA']))


# In[31]:

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


# In[32]:
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
    remove_unused_columns=False,
    label_names=[],
    dataloader_num_workers=4,
)

# %% train
# wandb.log({'phase': 'train'})
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
    eval_dataset=(
        datasets['testA']
    ),
#    compute_metrics=compute_metrics,
    callbacks=[train_logger],
)

trainer.can_return_loss = True

# In[36]:


trainer.train()