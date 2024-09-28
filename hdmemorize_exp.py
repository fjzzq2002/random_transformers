#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


from tqdm.auto import tqdm
arch = 'transformer'
train_steps = 21000
seed = 0
n_embd = 128
n_layer = 2
n_positions = 5
n_head = 4
device = 'cuda'
weight_decay = 0.0001
learning_rate = 1e-3 if 'arch' == 'transformer' else 5e-3
warmup_steps = 500
save_steps = 600
eval_steps = 100
logging_steps = 600
save_total_limit = 3
evaluation_strategy = "steps"
lr_scheduler_type = "cosine"
random_str = ''.join([random.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(8)])
task = 'hdmemorize_512_2_1.0'
calc_pca = False #True #False
tie_emb = False
weight_frozen = 1
dropout = 0.

# %% seed everything
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# # %% config from command line
exec(open('configurator.py').read()) # overrides from command line or config file
print(task,flush=True)

# guarantee a deterministic data generation process
random.seed(12345)

# %% specify the task
full_train_size = 0
if task.startswith('hdmemorize_'):
    task_split = task.split('_')
    if len(task_split) == 4:
        n, d, frac = int(task_split[1]), int(task_split[2]), float(task_split[3])
        vocab_size = n * d
        full_train_size = int(n ** d * frac)
        def taskA_gen():
            u = [[raw//(n**dd)%n+dd*n for dd in range(d)]+[random.randint(0,n-1)] for raw in range(n**d)]
            random.shuffle(u)
            return u[:full_train_size]
        print('haha')
assert full_train_size > 0
# everything fits!
per_device_train_batch_size = min(full_train_size,256*128)
per_device_eval_batch_size = min(full_train_size,256*128)


# In[3]:


# %% generate data
taskA_data = taskA_gen()
#taskB_data = taskB_gen()
# assert everything is within vocab size
for dd in taskA_data: assert all([0<=x<vocab_size for x in dd])
taskA_train = taskA_data
taskA_test = []
#taskB_train = taskB_train_all[:int(len(taskB_train_all)*fracB)]
print(f'Task A train: {len(taskA_train)}')
taskA_train_len = len(taskA_train)
#taskB_train_len = len(taskB_train)
taskA_test_len = len(taskA_test)
#taskB_test_len = len(taskB_test)


# In[ ]:





# In[4]:


# %% sanity check
# if train_style == 'mix':
#     assert taskA_train_len > 0# or taskB_train_len > 0
# elif train_style == 'finetune':
#     assert taskA_train_len > 0 and taskB_train_len > 0

# %% generate run description
run_description = task+' w'+str(n_embd)+' f'+str(weight_frozen)+' l'+str(n_layer)+' a'+str(arch)
# create folder task if not exists
if not os.path.exists(task):
    os.makedirs(task)
save_path = task+'/'+(random_str+' '+task+' '+' '+run_description).replace('(','_').replace(')','').replace(' ','_').replace(',','_')
print('Run',run_description)
print('Path',save_path)

# %% init wandb
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
print(config)
wandb.init(project='rand_transformer_vwfn', name=run_description, config=config)
wandb.run.log_code(".")

if arch == 'transformer':
    # %% define the model
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
else:
    import random
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence
    from tqdm.auto import tqdm


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
        self.input_ids = tokenized_data.clone()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx],
                "labels": self.input_ids[idx]}

# Load your custom tokenized dataset (assuming each example is a list of token IDs)
# Padding is currently not implemented
datasets = {'trainA': taskA_train}
datasets = {key: CustomDataset(tokenized_data=torch.tensor(val, dtype=torch.long)) for key, val in datasets.items() if len(val)}

print(datasets.keys())

for p,q in datasets.items():
    print(f'{p}: {len(q)}')


# In[5]:


# %% second train
#wandb.log({'phase': 'train'})
num_frozen = 0
num_trainable = 0

for n,p in model.named_parameters():
    if not any(x in n for x in (['wte','lm_head','wpe','encoder','decoder'])) and weight_frozen:
        print(n,p.shape,'frozen')
        p.requires_grad=False
    else:
        print('trainable:',n,p.shape)
    if p.requires_grad:
        num_trainable += p.numel()
    else:
        num_frozen += p.numel()
            
print(f'# parameters: {num_trainable} trainable, {num_frozen} frozen, {num_trainable+num_frozen} total')


# In[6]:


# %% some training utils
from transformers.trainer_callback import TrainerCallback
import wandb

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)[...,-2]
    corr = predictions == labels[...,-1]
    return {'accuracy': np.mean(corr)}

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


# In[7]:


training_args = TrainingArguments(
    num_train_epochs=train_steps,
    **training_args_dict
)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
#        print(inputs)
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # print(labels.shape,logits.shape)
        labels = labels[:,-1]
        logits = logits[:,-2]
        # print(labels.shape,logits.shape)
        # print(logits.view(-1, logits.shape[-1]).shape)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# train_logger = L2NormCallback('train')
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=(
        datasets['trainA']
    ),
    eval_dataset={'train_'+a:b for a,b in datasets.items()},
    compute_metrics=compute_metrics,
    # callbacks=[train_logger],
)


# In[ ]:


trainer.train()


# In[ ]:


78


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




