#Sample
import torch, os,tiktoken
from model import GPT
#----------------------------------------------------------------
init_from = "scatch"
out_dir = "outputs"
#sampling durining training
start = "<|endoftext|>"
num_samples = 10
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 1.0
top_k = 200 #retain onlt the first topo k topkens, the othe will have probabilities of 0
#system
seed = 42
device = "cuda"
dtype = "bfloat16"
compile = False
#----------------------------------------------------------------
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device_type = "cuda"
ptdtype = torch.bfloat16
ctx = torch.amp.autocast(device_type = device_type, dtype = ptdtype)

#model 
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    model = GPT(**checkpoint_model_args)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
model.eval()
model.to(device)

tokenizer = tiktoken.get_encoding("gpt2")
start_ids = tokenizer.encode(start, allowed_special={start})
print(start_ids)
x = torch.tensor(start_ids, dtype = torch.int64, device = device) [None, ...]
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature = temperature, top_k = top_k)
            print(tokenizer.decode(y[0].tolist()))
            print("------------------------------")

