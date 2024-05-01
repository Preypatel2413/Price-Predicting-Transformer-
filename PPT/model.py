import torch 
import os
import torch.nn as nn
import torch.nn.functional as F
from main import parse_json_data, feature_extract, normalize


n_embd = 64
n_layer = 8
n_head = 3
feature_size = 32
batch_size = 128
block_size = 180
dropout = 0.2
learning_rate = 3e-4

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AttentionHead(nn.Module):

    def __init__(self, head_size):
        super(AttentionHead, self).__init__()
        self.query_genertor = nn.Linear(n_embd, head_size)
        self.key_generator = nn.Linear(n_embd, head_size)
        self.value_computer = nn.Linear(n_embd, head_size)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedded_data):
        B,T,C = embedded_data.shape
        query = self.query_genertor(embedded_data)
        key = self.key_generator(embedded_data)

        wei = query @ key.transpose(-2,-1) * C**(-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        value = self.value_computer(embedded_data)

        out = wei @ value 

        return out


class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedded_data):
        out = torch.cat([h(embedded_data) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 2 * n_embd),
            nn.ReLU(),
            nn.Linear(2*n_embd, 2*n_embd),
            nn.ReLU(),
            nn.Linear(2*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd * 2
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class PPT(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Linear(feature_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, feature_size)

        self.apply(self._init_weights)

    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets = None):
        # print(idx.shape)
        B, T, C = idx.shape

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        inxd = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = inxd.shape
            # print(B, T, C)
            # print(logits.shape, targets.shape)
            inxd = inxd.view(B*T, C)
            targets = targets.view(B*T, C)
            loss = F.mse_loss(inxd, targets)

        return inxd, loss
    

    def generate(self, idx, max_new_prediction):
        
        for _ in range(max_new_prediction):
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)

            logits = logits[:,-1:]

            idx_next = logits

            # print(idx.shape, idx_next.shape)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    

# model = PPT()

# m = model.to(device)

# print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# torch.save({
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
# }, 'ppt_cp.pt')





# a = {"status":"success","data":{"candles":[["2023-07-26T15:29:00+05:30",194.3,194.3,194.3,194.3,0,0],["2023-07-26T15:28:00+05:30",194.3,194.3,194.3,194.3,0,0],["2023-07-26T15:27:00+05:30",195.05,195.05,194.3,194.3,200,2480],["2023-07-26T15:26:00+05:30",195.1,195.1,195.1,195.1,240,2480],["2023-07-26T15:25:00+05:30",195.9,195.9,195.9,195.9,0,0],["2023-07-26T15:24:00+05:30",195.9,195.9,195.9,195.9,40,2480],["2023-07-26T15:23:00+05:30",197,197,197,197,0,0],["2023-07-26T15:14:00+05:30",196.35,196.35,196.35,196.35,0,0],["2023-07-26T15:13:00+05:30",196.35,196.35,196.35,196.35,0,0],["2023-07-26T15:12:00+05:30",196.35,196.35,196.35,196.35,0,0],["2023-07-26T15:11:00+05:30",193.75,196.35,193.75,196.35,80,2560],["2023-07-26T15:10:00+05:30",199.95,199.95,199.95,199.95,0,0],["2023-07-26T15:09:00+05:30",199.95,199.95,199.95,199.95,40,2520],["2023-07-26T15:08:00+05:30",194,194,194,194,0,0],["2023-07-26T15:07:00+05:30",194,194,194,194,0,0],["2023-07-26T15:06:00+05:30",194,194,194,194,0,0],["2023-07-26T15:05:00+05:30",194,194,194,194,0,0],["2023-07-26T15:04:00+05:30",194,194,194,194,40,2480],["2023-07-26T15:03:00+05:30",204.1,204.1,204.1,204.1,0,0],["2023-07-26T15:02:00+05:30",202.1,204.1,202.1,204.1,80,2480]]}}
# B = parse_json_data(a)
# C = feature_extract(B)
# data = torch.tensor(C, dtype=torch.float)
# n = int(0.9*len(data))
# train_data = data[:n]
# val_data = data[int(2*n/3):]

# def get_batch(split):
#     block_size = 5
#     data = train_data if(split == "train") else val_data
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     x = torch.stack([data[i:i+block_size] for i in ix])
#     y = torch.stack([data[i+1:i+block_size+1] for i in ix])
#     x, y = x.to(device), y.to(device)

#     return x, y

# def estimate_loss():
#     out = {}
#     model.eval()
#     for split in ['train', 'val']:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             X, Y = get_batch(split)
#             logits, loss = model(X, Y)
#             losses[k] = loss.item()
        
#         out[split] = losses.mean()
    
#     model.train()
#     return out

# print("Training start")
# for iter in range(max_iters):

#     if iter % eval_interval == 0 or iter == max_iters -1:
#         losses = estimate_loss()
#         print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

#     xb, yb = get_batch('train')
#     logits, loss = model(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

# print("Training End")


# a = {"status":"success","data":{"candles":[["2023-07-26T15:29:00+05:30",194.3,194.3,194.3,194.3,0,0],["2023-07-26T15:28:00+05:30",194.3,194.3,194.3,194.3,0,0],["2023-07-26T15:27:00+05:30",195.05,195.05,194.3,194.3,200,2480],["2023-07-26T15:26:00+05:30",195.1,195.1,195.1,195.1,240,2480],["2023-07-26T15:25:00+05:30",195.9,195.9,195.9,195.9,0,0],["2023-07-26T15:24:00+05:30",195.9,195.9,195.9,195.9,40,2480],["2023-07-26T15:23:00+05:30",197,197,197,197,0,0]]}}
# B = parse_json_data(a)
# C = feature_extract(B)
# context = (torch.tensor([C], dtype=torch.float)).to(device)

# print(context, context.shape)
# prd = m.generate(context, max_new_prediction=30)
# print(prd.shape)
# prd = prd.tolist()
# for i in range(len(prd[0])):
#     print(prd[0][i][:8])
