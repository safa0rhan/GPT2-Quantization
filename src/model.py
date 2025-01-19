import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append('.')
import time

# print(torch.cuda.is_available())  # Should print True for CUDA-enabled systems
# print(torch.cuda.device_count())  # Should match the number of GPUs

@dataclass
class GPTConfig:
    block_size: int = 1024    # max sequence length
    vocab_size: int = 50257   # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12         # number of layers
    n_head: int = 12          # number of heads
    n_embd: int = 768         # embedding dimension
    model_path: str = "/home/hpc/ihpc/ihpc135h/dd2375/gpt-2"         # path to the model weights


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
            persistent=False
        )

    def forward(self, x):
        B, T, E = x.shape  # batch, time, embedding_dim
        qkv = self.c_attn(x)  # shape (B, T, 3*E)
        q, k, v = qkv.split(E, dim=2)  # each shape (B, T, E)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        scaled_score = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scaled_score = scaled_score.masked_fill(
            self.tril[:, :, :T, :T] == 0, float('-inf')
        )

        attn_wei = F.softmax(scaled_score, dim=-1)
        out = attn_wei @ v  # shape (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, E)  # back to (B, T, E)
        out = self.c_proj(out)
        return out
        
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Hugging Face GPT-2 style
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act    = nn.GELU() # activation function

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x

class GPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(self.config.vocab_size, self.config.n_embd),
            'wpe': nn.Embedding(self.config.block_size, self.config.n_embd),
            'h': nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),
        })

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        #print(f"Model block size: {self.config.block_size}")
        #print(B, T)
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(config.model_path)
        sd_hf = model_hf.state_dict()

        # remove the tril buffer
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.tril')]
        # remove attn.bias if needed
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model



num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained('gpt2')
model.eval()
model.to('cuda')

# Encoding
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("/home/hpc/ihpc/ihpc135h/dd2375/gpt-2")
text = "Hello, I'm a language model,"
tokens = tokenizer.encode(text)
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
tokens = tokens.to('cuda')

torch.manual_seed(42)
torch.cuda.manual_seed(42)

warmup_iter = 50
iter_ = 0
while iter_ < warmup_iter:
    with torch.no_grad():
        _ = model(tokens)
    torch.cuda.synchronize()
    iter_ += 1

# -----------------------------------------------------------------
# Timing variables
# -----------------------------------------------------------------
time_to_first_token = None
inter_token_latencies = []
start_time = time.time()
prev_time = start_time

# We also might want to do an initial CUDA synchronize just before:
torch.cuda.synchronize()
start_time = time.time()
prev_time = start_time

# Generation loop
while tokens.size(1) < max_length:
    with torch.no_grad():
        # make sure x does not exceed the model's block_size
        if tokens.size(1) > model.config.block_size:
            tokens = tokens[:, -model.config.block_size:]
        # forward pass
        logits = model(tokens)
        # we only sample from the last token
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 5, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        xcol = xcol.to(tokens.device)
        tokens = torch.cat((tokens, xcol), dim=1)
    
    # Sync + measure times
    torch.cuda.synchronize()
    now_time = time.time()
    
    if time_to_first_token is None:
        # The first time we successfully generate a new token
        time_to_first_token = now_time - start_time
    else:
        # inter-token from last step
        inter_token_latencies.append(now_time - prev_time)

    prev_time = now_time

torch.cuda.synchronize()
total_time = time.time() - start_time

# -----------------------------------------------------------------
# Print timing
# -----------------------------------------------------------------
throughput = tokens.size(1) / total_time
print(f"Throughput: {throughput:.2f} tokens/second")
print(f"Time to first token: {time_to_first_token:.4f} seconds")
print(f"Total time to generate up to length {tokens.size(1)}: {total_time:.4f} seconds")

# inter-token stats
if inter_token_latencies:
    avg_inter = sum(inter_token_latencies) / len(inter_token_latencies)
    print(f"Avg inter-token latency: {avg_inter:.4f} seconds")
    print(f"Per-token latencies: {inter_token_latencies}")
else:
    print("No inter-token latencies found (only 1 token was generated).")

# -----------------------------------------------------------------
# Decoding output
# -----------------------------------------------------------------
for i in range(num_return_sequences):
    out_toks = tokens[i, :max_length].tolist()
    decoded = tokenizer.decode(out_toks)
    print(">", decoded)



"""
# Inference loop
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 5, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

# Decoding output
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = tokenizer.decode(tokens)
    print(">", decoded)

print("--------------------------------- Done! :) Done!  Done!  Done!  Done! ---------------------------------")


# Encode input text
tokens = tokenizer.encode(text)
tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to('cuda')

# Generate text
with torch.no_grad():
    for _ in range(100):
        logits = model(tokens)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, 1)
        tokens = torch.cat((tokens, ix), dim=1)

# Decode output text
tokens = tokens.squeeze().tolist()
decoded = tokenizer.decode(tokens)
print(decoded)
print("--------------------------------- Done! :) Done!  Done!  Done!  Done! ---------------------------------")
"""