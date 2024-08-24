import os, time, math, pickle, torch, argparse, yaml
import numpy as np
from model import GPT

#Getting the config from the command line
parser = argparse.ArgumentParser()
parser.add_argument("--config", type = str, required = True, help = "config path")

args = parser.parse_args()
config_path = args.config
with open(config_path) as file:
    config = yaml.safe_load(file)

#logging parameters
logging_params = config["logging_params"]
out_dir = logging_params["out_dir"]
always_save_checkpoint = logging_params["always_save_checkpoint"]
eval_iters = logging_params["eval_iters"]
eval_interval = logging_params["eval_interval"]
log_interval = logging_params["log_interval"]
init_from = logging_params["init_from"]
del logging_params 

#dataset parameters
dataset = config["dataset"]
print(dataset)
data_dir = r"C:\Users\Gianl\Desktop\gpt-repository\data"
meta = data_dir + os.sep + dataset + "-meta.pkl" if os.path.exists(data_dir + os.sep + dataset + "-meta.pkl") else None
if meta:
    with open(meta, "rb") as f:
        meta = pickle.load(f)
        decode = meta["itos"]
        encode = meta["stoi"]
        vocab_size = meta["vocab_size"]
else:
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = 50304
#getting vocab size
print("Vocab size is:", vocab_size)

#model
model_params = config["model_params"]
n_layers = model_params["n_layers"]
n_heads= model_params["n_heads"]
emb = model_params["emb"]
dropout = model_params["dropout"]
del model_params

# training params
training_params = config["training_params"]
gradient_accomulation_iter = training_params["gradient_accomulation_iter"]
batch_size = training_params["batch_size"]
context = training_params["context"]
#sampling during training

sample_duringtraining = training_params["sample_duringtraining"]
start = training_params["start"] 
num_samples = training_params["num_samples"]
max_new_tokens = training_params["max_new_tokens"]
del training_params

#Optimizier
optimizer_params = config["optimizer_params"] 
max_lr = optimizer_params["max_lr"]
max_iters = optimizer_params["max_iters"]
weight_decay = optimizer_params["weight_decay"]
b1 = optimizer_params["b1"]
b2 = optimizer_params["b2"]
grad_clip = optimizer_params["grad_clip"] # clipping the norm of the gradient to this value
del optimizer_params

#learning rate scheduler
scheduler_params = config["scheduler_params"]
decay_lr = scheduler_params["decay_lr"]
warmup_iters = scheduler_params["warmup_iters"]
lr_decay_iters = scheduler_params["lr_decay_iters"] # iterations of cosine decay, should be similar to the max iterations per Karpathy
min_lr = scheduler_params["min_lr"] # minimum learning rate of the decay, should be similar to lr/10 per Karpathy
del scheduler_params
del config
# system
device = "cuda"
dtype = "bfloat16"if torch.cuda.is_available() and torch.cuda.is_bf16_supported else "float16"
torch.manual_seed(42)
pdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type = device, dtype = pdtype)
model_args = dict(n_layers=n_layers, n_heads=n_heads, emb=emb, context=context, vocab_size=vocab_size, dropout=dropout)

#initialization of the model
iter_num = 0
best_val_losses = 1e9

if init_from == "scratch":
    #init the model from scratch
    print("Initializing a new model from scratch")
    model = GPT(**model_args)
elif init_from == "resume":
    print(f"Resuming from {out_dir}")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location = device)
    checkpoint_model_args = checkpoint["model_args"]
    for k in ['n_layers', 'n_heads', 'emb', 'context', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    model = GPT(**model_args)
    state_dict = checkpoint["model"]
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_losses = checkpoint["best_val_losses"]
model.to(device)



def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        #already tokenized data
        data = np.memmap(os.path.join(data_dir, dataset + '-train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, dataset + '-val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - context, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+context]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+context]).astype(np.int64)) for i in ix])
    
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


scaler = torch.cuda.amp.GradScaler(enabled = (dtype == "float16"))
#optimizer
optimizer = model.configure_optimizers(weight_decay, max_lr, (b1, b2), device_type=device)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None #flush the memory
#helps get an accurate loss using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            data, labels = get_batch(split)
            with ctx:
                logits, loss = model(data, labels)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
# learning rate scheduler (cosine + warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return max_lr * (it+1) / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

if sample_duringtraining and not meta:
    tokenizer = tiktoken.get_encoding("gpt2")
    start_ids = tokenizer.encode(start, allowed_special={start})
    x = torch.tensor(start_ids, dtype = torch.int64, device = device).unsqueeze(0)
if sample_duringtraining and meta:
    print(start)
    start_ids = encode[start]
    x = torch.tensor([start_ids], dtype = torch.int64, device = device).unsqueeze(0)
    
def sampling(x):
    model.eval()
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens)
                if not meta:
                    print(tokenizer.decode(y[0].tolist()))
                else:
                    text = []
                    for i in y[0].tolist():
                        text.extend(decode[i])
                    print("".join(text))
                    text = []
                print("------------------------------")
    model.train()
    
#TRAINING LOOP
data, labels = get_batch("train")
t0 = time.time()
num_of_tokens_per_step = gradient_accomulation_iter * batch_size * context
train_losses = []
val_losses = []
while True:
    lr = get_lr(iter_num) if decay_lr else max_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    if iter_num % eval_interval == 0 :
        losses = estimate_loss()
        val_losses.append(losses["val"])
        print(f"val | step {iter_num}| train loss {losses['train']:.4f}| val loss {losses['val']:.4f}")
        if sample_duringtraining:
            sampling(x)
        if losses["val"] < best_val_losses or always_save_checkpoint:
            best_val_losses = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model" : model.state_dict(),
                    "optimizer" : optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_losses": best_val_losses,
                }    
                print(f"save checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            with open(out_dir + os.sep + "train_losses.pkl", "wb") as file:
                    pickle.dump(train_losses, file)
            with open(out_dir + os.sep +  "val_losses.pkl", "wb") as file:
                    pickle.dump(val_losses, file)
    #gradient accumulation        
    for micro_step in range(gradient_accomulation_iter):
        with ctx:
            logits, loss = model(data, labels)
            loss = loss / gradient_accomulation_iter
        data, labels = get_batch("train")
        scaler.scale(loss).backward()
    #clip the gradient
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    t1 = time.time()
    dt = (t1-t0)
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accomulation_iter
        train_losses.append(lossf)
        completion_time = (max_iters - iter_num)*dt/3600
        print(f"step {iter_num}| loss {lossf:.4f}| time {dt:.2f} sec| lr: {lr:.6f}| num_of_tokens_processed_per_step: {num_of_tokens_per_step}| Expected time left : {completion_time:.1f} hrs")    
    iter_num +=1
    if iter_num > max_iters:
        with open(out_dir + os.sep + "train_losses.pkl", "wb") as file:
            pickle.dump(train_losses, file)
        with open(out_dir + os.sep +  "val_losses.pkl", "wb") as file:
            pickle.dump(val_losses, file)
        break
    
    
    
