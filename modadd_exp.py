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


fracA = 0.95
train_steps = 5000
seed = 0 #random.randint(0,10**9)
n_embd = 256
n_layer = 2
n_head = 4
n_positions = 5
device = 'cuda'
weight_decay = 0.001
learning_rate = 1e-3
warmup_steps = 500
save_steps = 600
eval_steps = 50
logging_steps = 600
save_total_limit = 3
evaluation_strategy = "steps"
lr_scheduler_type = "cosine"
random_str = ''.join([random.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(8)])
task = 'modadds_199'
calc_pca = False #True #False
tie_emb = False
weight_frozen = 1

# # %% config from command line
exec(open('configurator.py').read()) # overrides from command line or config file
print(task,flush=True)

# guarantee a deterministic data generation process
random.seed(12345)

# %% specify the task
full_train_size = 0
if task.startswith('modadds_'):
    task_split = task.split('_')
    if len(task_split) == 2:
        P = int(task_split[1])
        vocab_size = P
        def taskA_gen():
            return [[a,b,(a+b)%P] for a in range(P) for b in range(P)]
        full_train_size = int(P*P*fracA)
        print('haha')

assert full_train_size > 0
# everything fits!
per_device_train_batch_size = 4000
per_device_eval_batch_size = 4000

# %% generate data
taskA_data = taskA_gen()
# assert everything is within vocab size
for d in taskA_data: assert all([0<=x<vocab_size for x in d])
# shuffle
random.shuffle(taskA_data)
# split into train and test for both tasks, first {full_train_size} used for training
taskA_train = taskA_data[:full_train_size]
taskA_test = taskA_data[full_train_size:]
assert len(taskA_test) > 0 #and len(taskB_test) > 0
print(f'Task A train: {len(taskA_train)} (of {len(taskA_train)}), test: {len(taskA_test)}')
taskA_train_len = len(taskA_train)
taskA_test_len = len(taskA_test)

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
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging
print(config)
wandb.init(project='rand_transformer_vwfn_modadd', name=run_description, config=config)
wandb.run.log_code(".")

# %% define the model
config = GPT2Config(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_layer=n_layer,
    n_head=n_head,
    tie_word_embeddings=tie_emb,
    n_positions=n_positions,
    resid_pdrop = 0.0,
    embd_pdrop = 0.0,
    attn_pdrop = 0.0
)
model = GPT2LMHeadModel(config).to(device)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")


# %% build dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data.clone()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx],
                "labels": self.input_ids[idx],
                "attention_mask": torch.full_like(self.input_ids[idx],1)}

# Load your custom tokenized dataset (assuming each example is a list of token IDs)
# Padding is currently not implemented
datasets = {'trainA': taskA_train, 'testA': taskA_test}
datasets = {key: CustomDataset(tokenized_data=torch.tensor(val, dtype=torch.long)) for key, val in datasets.items() if len(val)}

print(datasets.keys())

for p,q in datasets.items():
    print(f'{p}: {len(q)}')

# %% some training utils
from transformers.trainer_callback import TrainerCallback
import wandb

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)[...,-2]
    corr = predictions == labels[...,-1]
    return {'accuracy': np.mean(corr)}

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

train_logger = L2NormCallback('train')
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=(
        datasets['trainA']
    ),
    eval_dataset={'train_'+a:b for a,b in datasets.items()},
    compute_metrics=compute_metrics,
    callbacks=[train_logger],
)

trainer.train()