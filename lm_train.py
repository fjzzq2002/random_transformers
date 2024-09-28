#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Ref:
# https://github.com/huggingface/transformers/blob/eb5bdcdfa51f743887ee1d9c7f230444d7a8b23c/examples/pytorch/language-modeling/run_mlm_no_trainer.py#L433
# 

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


train_steps = 5 #0 #000
seed = 44
n_embd = 512
n_layer = 4
n_head = 8
device = 'cuda'
weight_decay = 0.1
learning_rate = 6e-4
warmup_steps = 500
save_steps = 250
eval_steps = 250
logging_steps = 250
save_total_limit = 3
evaluation_strategy = "steps"
lr_scheduler_type = "cosine"
random_str = ''.join([random.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(8)])
calc_pca = False #True #False
tie_emb = False
weight_frozen = 0
task = 'tinystoriesclm'
per_device_train_batch_size = 20
per_device_eval_batch_size = 20

# # %% config from command line
exec(open('configurator.py').read()) # overrides from command line or config file
print(task,flush=True)

# ### Preprocess done...

# In[2]:


from datasets import load_from_disk
lm_datasets = load_from_disk('tinystories')
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("tinystories", max_length=512)


# %% generate run description
run_description = task+' w'+str(n_embd)+' f'+str(weight_frozen)
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
import os
os.environ["WANDB_MODE"] = "offline"
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
print(config)
wandb.init(project='rand_transformer_vwf', name=run_description, config=config)
wandb.run.log_code(".")

# %% define the model
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_embd=n_embd,
    n_layer=n_layer,
    n_head=n_head,
    tie_word_embeddings=tie_emb,
)
model = GPT2LMHeadModel(config).to(device)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")


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
training_args = TrainingArguments(
    num_train_epochs=train_steps,
    **training_args_dict
)


# In[11]:


import transformers
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets['train'],
    eval_dataset=lm_datasets['validation'],
    tokenizer=tokenizer,
    # Data collator will default to DataCollatorWithPadding, so we change it.
    data_collator=transformers.default_data_collator,
)


# In[12]:


# %% some training utils
from transformers.trainer_callback import TrainerCallback
import wandb

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)[...,-2]
    corr = predictions == labels[...,-1]
    return {'accuracy': np.mean(corr)}


# In[ ]:


# %% second train
num_frozen = 0
num_trainable = 0

for n,p in model.named_parameters():
    if weight_frozen==1 and not any(x in n for x in ['wte','lm_head','wpe']):
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

trainer.train()