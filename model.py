import torch, math, os, time
import torch.nn as nn
import torch.nn.functional as F

class Feed_Forward(nn.Module):
    def __init__(self, context, emb,  dropout):
        super().__init__()
        self.lin1 = nn.Linear(emb, 4 * emb, bias = False) # hidden dimensionality of 4, no bias
        self.act1 = nn.GELU() #using the gelu activation function
        self.drop= nn.Dropout(dropout)
        self.output_projection = nn.Linear(4 * emb, emb, bias = False)

    def forward(self, x: torch.Tensor):
        x = self.act1(self.lin1(x))
        x = self.drop(self.output_projection(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,  n_heads: int, context: int, emb: int, dropout: float = 0.0):
        super().__init__()
        assert emb % n_heads == 0, "embedding dimensionality must be a multiple of n_heads"
        
        self.emb = emb
        self.nh = n_heads
        self.dropout = dropout
        self.projection = nn.Linear(emb, emb*3) 
        self.output_projection = nn.Linear(emb, emb)
        self.output_dropout = nn.Dropout(dropout)
        self.flash = hasattr(nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("flesh is not working")
            self.register_buffer("causal_mask", torch.tril(torch.ones(context, context)).view(1,1, context, context))
            self.att_drop = nn.Dropout(dropout)
            
    def get_attention_scores(self, query: torch.Tensor, keys: torch.Tensor, values: torch.tensor):
        if self.flash:
            out = F.scaled_dot_product_attention(query, keys, values, attn_mask=None, dropout_p=self.dropout if self.training else 0,
                                                                    is_causal=True)
        else: 
            scale = math.sqrt(self.emb)
            att_scores = (query @ keys.transpose(-2,-1))/scale 
            B, NH, C, E = query.shape
            att_scores.masked_fill_(self.causal_mask[:,:,:C, :C] == 0, float('-inf'))
            att_scores = att_scores.softmax(y, dim = -1)
            att_scores = self.att_drop(att_scores)
            out = att_scores @ values
        return out
    
    def forward(self, x: torch.Tensor):
        B, C, E = x.shape
        #splitting in queries, keys, values
        queries, keys, values = self.projection(x).split(self.emb, dim = 2) # batch, context, emb * 3 --> queries, keys, values of batch, context, emb
        queries = queries.view(B, C, self.nh, E // self.nh).transpose(1,2) # b, nh, c, emb
        keys = keys.view(B, C, self.nh, E // self.nh).transpose(1,2)
        values = values.view(B, C, self.nh, E // self.nh).transpose(1,2)
        x = self.get_attention_scores(queries, keys, values) # b, nh, c, c
        x = x.transpose(1,2).contiguous().view(B, C, E) #recomposing the vector
        x = self.output_dropout(self.output_projection(x)) #final linea layer and dropout
        return x    
    
class TransformerBlock(nn.Module):
    def __init__(self, n_heads: int, context: int, emb: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb, bias = False)
        self.MultiHead = MultiHeadAttention(n_heads, context, emb, dropout)
        self.ln2 = nn.LayerNorm(emb, bias = False)
        self.FeedForward = Feed_Forward(context, emb,dropout)
        
    def forward(self, x: torch.Tensor):
        #Composing a single block with residual connection and layer normalization
        x = x + self.MultiHead(self.ln1(x))
        x = x + self.FeedForward(self.ln2(x)) #(B,C,D)
        return x    
# Model
class GPT(nn.Module):
    def __init__(self, n_layers : int, n_heads: int, context: int, emb: int, bias = False, vocab_size: int = 50304, dropout: float = 0.0):
        super().__init__()
        
        self.context = context

        self.emb = nn.Embedding(vocab_size, emb)
        self.pos = nn.Embedding(context, emb)
        self.drop = nn.Dropout(dropout)
        
        self.TransformerBlocks = nn.ModuleList(
            [TransformerBlock(n_heads, context, emb, dropout)
            for _ in range(n_layers)])
        self.fln = nn.LayerNorm(emb, bias = False)
        self.generator = nn.Linear(emb, vocab_size, bias = False) #final layer
    # same parameter for the embedding and the final layer to reduce the total parameters
        
        self.emb.weight =  self.generator.weight
        
        self.apply(self._init_weights)
    # special normalization for residuals, as done in the GPT2 paper
        for pn, p in self.named_parameters():
                if pn.endswith('output_projection.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layers))
                    
    # starting the weights as in gpt2                 
    def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)   
                
    # total number of parameters with or without position's weights            
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.pos.weight.numel()
        return n_params          
                
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Getting parameters with requires_grad = True
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer    

    def forward(self, x: torch.Tensor, targets = None):
        device = x.device
        B, C = x.shape
        embs = self.emb(x) #batch, seq, emb
        pos = self.pos(torch.tensor([i for i in range(C)], dtype = torch.int64, device=device)) # position embedding
        x = self.drop(embs + pos)
        for Transf in self.TransformerBlocks:
            x = Transf(x)
        x = self.fln(x)
        
        if targets is not None:
            logits = self.generator(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.generator(x[:, [-1], :]) 
            loss = None    
        return logits, loss
    
    #generation from previous tokens
    @torch.no_grad()
    def generate(self, idx, max_tokens, temperature: float= 1.0, top_k: int = 200):
        B = idx.shape[0]
        for _ in range(max_tokens):
            new_idx = idx[:, -self.context:]
            logits, _ = self(new_idx)
            logits = logits[:,-1,:] /temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:,[-1]]] = float("-inf")
            probs = logits.softmax(dim = -1)
            next_toks = torch.multinomial(probs, num_samples =1) 
            for i in range(B):
                while next_toks[i].item() >= 50256:
                    probs[i, next_toks[i]] = 0  # Remove the invalid token from the probability distribution
                    next_toks[i] = torch.multinomial(probs[i], num_samples=1)        
            idx = torch.cat((idx, next_toks), dim = 1)
        return idx
