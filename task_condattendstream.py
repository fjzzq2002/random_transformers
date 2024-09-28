train_steps = 250
seed = 43
n_embd = 16
n_layer = 2
n_head = 4
n_positions = 100
weight_decay = 0.001
warmup_steps = 500
save_steps = 600
eval_steps = 50
logging_steps = 600
vocab_size = 256
train_size = 40000
eval_size = 4000
dropout = 0.
per_device_train_batch_size = 1000
per_device_eval_batch_size = 1000

task_split = task.split('_')
max_len = int(task_split[1])
def gen_single():
    seq_len = random.randint(1,max_len)
    keys = list(range(max_len))
    random.shuffle(keys)
    keys = keys[:seq_len]
    values = [random.randint(1,vocab_size//2-1) for _ in range(seq_len)]
    query = random.choice(keys)
    l = sum(([a+vocab_size//2,b] for a,b in zip(keys,values)),[])+[query+max_len+vocab_size//2]
    l = l + [values[keys.index(query)]]
    l += [0] * (max_len*2+3-len(l))
    assert max(l)<vocab_size and min(l)>=0
    return l
def taskA_gen():
    tasks = []
    for i in range(eval_size):
        tasks.append(gen_single())
    return tasks

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    wanted = labels[...,:-1]>=max_len+vocab_size//2
    predictions = np.argmax(predictions, axis=-1)[...,:-1]
    corr = (predictions == labels[...,1:])*wanted
    return {'accuracy': np.sum(corr)/np.sum(wanted)}


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        wanted = labels[:,:-1].detach().flatten()>=max_len+vocab_size//2
        logits = logits[:,:-1]
        labels = labels[:,1:]
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        ws = wanted.sum()
        loss = (loss * wanted).sum() / ws
        return (loss, outputs) if return_outputs else loss