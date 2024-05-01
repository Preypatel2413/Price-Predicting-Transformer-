from model import *
import time
import json

max_iters = 10#
eval_interval = 100#
eval_iters = 5#


model = PPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


checkpoint = torch.load('ppt_cp.pt')

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# print("initial model eval : ", model.eval())
m = model.to(device)

def get_batch(split, obs_size = 30):
    data = train_data if(split == "train") else val_data
    ix = torch.randint(len(data) - obs_size, (batch_size,))
    x = torch.stack([data[i:i+obs_size] for i in ix])
    y = torch.stack([data[i+1:i+obs_size+1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y

def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        
        out[split] = losses.mean()
    
    model.train()
    return out

print(device)



i = 0

for d in range(127, 500):

    with open(f'Data/data_ ({d}).json', 'r') as f:
        json_data = json.load(f)

    parsed_data = parse_json_data(json_data)
    feature_vec = feature_extract(parsed_data)
    norm_data = normalize(feature_vec)
    data = (torch.tensor(norm_data, dtype=torch.float)).to(device)

    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[int(2*n/3):]

    stime = time.time()
    print("Training start:", d, len(feature_vec))
    for iter in range(max_iters):

        for b_size in range(20, 185, 5):
            if i % eval_interval == 0 or (iter == max_iters - 1 and b_size==180):
                losses = estimate_loss()
                print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb, yb = get_batch('train', b_size)
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            i+=1

    print("Training End.    time: ", time.time() - stime)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'ppt_cp.pt')

    if(d % 40==0):
        torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'checkpoints/ppt_checkpoints_{d}.pt')

    print("model saved \n")