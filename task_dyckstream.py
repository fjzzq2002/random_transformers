train_steps = 250
seed = 43
n_embd = 16
n_layer = 2
n_head = 4
n_positions = 80
weight_decay = 0.001
warmup_steps = 500
save_steps = 600
eval_steps = 50
logging_steps = 600
vocab_size = 4 #!!!
train_size = 40000
eval_size = 4000
dropout = 0.
per_device_train_batch_size = 1000
per_device_eval_batch_size = 1000

task_split = task.split('_')
max_len = int(task_split[1])
def gen_single():
    # 0: EOF  1: (   2: )   3: ?
    assert 3 == vocab_size-1  # position of ?
    def gen_balanced(t):
        assert t>=0
        if t==0: return ''
        if t==1: return '()'
        if random.randint(0,1): return '('+gen_balanced(t-1)+')'
        u = random.randint(1,t-1)
        return gen_balanced(u)+gen_balanced(t-u)
    mutate = random.randint(0,3)
    if random.randint(0,2)==0:
        seq_len = random.randint(1,max_len*2)
        l = ''.join([random.choice('()') for _ in range(seq_len)])
    else:
        # random balanced
        l = gen_balanced(random.randint(1,max_len))
        if random.randint(0,1):
            mutate = 0  # 1/3 chance of correct + no mutation
    if mutate&1:
        rep = 1
        while random.randint(0,1): rep += 1
        for _ in range(rep):
            p = random.randint(0,len(l)-1)
            q = random.randint(0,len(l)-1)
            if p>=q: continue
            # swap l[p],l[q]
            l = l[:p]+l[q]+l[p+1:q]+l[p]+l[q+1:]
    if mutate&2:
        rep = 1
        while random.randint(0,1): rep += 1
        if random.randint(0,2):
            rep *= 2
        for _ in range(rep):
            p = random.randint(0,len(l)-1)
            # flip l[p]
            l = l[:p]+({'(':')',')':'('}[l[p]])+l[p+1:]
    def is_balanced(l):
        s = 0
        for c in l:
            if c=='(':
                s += 1
            elif c==')':
                if s<=0: return False
                s -= 1
        return s==0
    # cnt_balanced += is_balanced(l)
    l = l + '?' + (')' if is_balanced(l) else '(')
    arr = [{'(':1,')':2,'?':3}[c] for c in l]
    assert max_len*2+3-len(arr)>=0
    arr += [0] * (max_len*2+3-len(arr))
    return arr
def taskA_gen():
    tasks = []
    for i in range(eval_size):
        tasks.append(gen_single())
    return tasks

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    wanted = labels[...,:-1]==vocab_size-1
    predictions = np.argmax(predictions, axis=-1)[...,:-1]
    corr = (predictions == labels[...,1:])*wanted
    return {'accuracy': np.sum(corr)/np.sum(wanted)}


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        wanted = labels[:,:-1].detach().flatten()==vocab_size-1
        logits = logits[:,:-1]
        labels = labels[:,1:]
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        ws = wanted.sum()
        loss = (loss * wanted).sum() / ws
        return (loss, outputs) if return_outputs else loss