#Sample
import torch, os,tiktoken, pickle
from model import GPT
#----------------------------------------------------------------
init_from = "resume"
out_dir = "outputs"
#sampling method
start = "\n"
num_samples = 3
max_new_tokens = 250 # number of tokens generated in each sample
temperature = 0.5
top_k = 200 #retain onlt the first topo k topkens, the othe will have probabilities of 0
#system
seed = 42
device = "cuda"
dtype = "bfloat16"
compile = False
#utils
data_dir = r"C:\Users\Gianl\Desktop\gpt-repository\data"
dataset = "dailymails"
meta = data_dir + os.sep + dataset + "-meta.pkl" if os.path.exists(data_dir + os.sep + dataset + "-meta.pkl") else None
#----------------------------------------------------------------
torch.manual_seed(seed)
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
if meta:
    with open(meta, "rb") as f:
        meta = pickle.load(f)
        decode = meta["itos"]
        encode = meta["stoi"]
        vocab_size = meta["vocab_size"]
        start_ids = encode[start]
else:
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    start_ids = tokenizer.encode(start, allowed_special={start})
    
x = torch.tensor(start_ids, dtype = torch.int64, device = device).repeat(num_samples,1)

with torch.no_grad():
    with ctx:
            y = model.generate(x, max_new_tokens, temperature = temperature, top_k = top_k)
            if not meta:
                for i in range(y.shape[0]):
                    
                    print(tokenizer.decode(y[i].tolist()))
                    print("------------------------------")
            else:
                text = []
                for i in range(y.shape[0]):
                    for idx in y[i].tolist():
                        text.extend(decode[idx])
                    print("".join(text))
                    text = []
                    print("------------------------------")

