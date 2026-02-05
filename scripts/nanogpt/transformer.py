""" Script to implement Transformer based language model"""

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# hyperparameters 
batch_size = 32
block_size = 8
max_iterations = 3000
learning_rate = 1e-2
eval_iters = 200
eval_interval = 300 
device = "cuda" if torch.cuda.is_available() else 'cpu'
n_embd = 32 # no of embedding dimention
torch.manual_seed(1337)
# -------------------------------------

# read the data 
with open("/Users/rahulkrish/Desktop/myrepos/deeplearning/data/input.txt", "r", encoding= 'utf-8')as f:
    text = f.read()
print(f"No of charcters = {len(text)}")
print(f"No of line = {len(text.splitlines())}")

# identify vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"unique character: {''.join(chars)}")
print(f"vocab_size: {vocab_size}")
# create a mapping from chars to int and vise-versa
stoi = {s:i for i,s in enumerate(chars)}
itos = {v:k for k,v in stoi.items()}

# encoder and decoder 
encode = lambda  s:[stoi[c] for c in s] #encoder: take a string, output a list of integer
decode = lambda  l: ''.join([itos[i] for i in l]) # decode: takes a list of indexes and output a string


# train and val split 
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Length of train data: {len(train_data)}")
print(f"Length of validation data: {len(val_data)}")


# get batch 
def get_batch(split):
    """
    Get a mini batch from specified data split
    input: "train" or "val"
    output: x, y 
    y is x moved to right by an index 
    x.shape -> [4,8]
    y.shape -> [4,8]
    """
    data = train_data if split =="train" else val_data
    ix = torch.randint(len(data) - block_size-1,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) # move batch to device
    return x,y

# biagram language model 
class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits of the next token from a look up table
        # nn.Embedding -> A simple lookup table that stores embeddings of a fixed dictionary and size.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx,targets=None):
        B,T = idx.shape
        # idx and targets are of shape (B,T)
        x = self.token_embedding_table(idx) # (B,T,n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            # pytorch cross entropy expects second dimention as the channel
            logits = logits.view(B*T , C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is B,T array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions 
            logits, loss = self(idx) # (B, T, C) # self here runs the def forward 
            # get predictions for the last time step 
            logits = logits[:, -1, :] #(B,C) pluck the last time step
            # apply soft max to get the probabilities 
            prob = F.softmax(logits, dim = -1) #(B, C)
            # sample from the distibution 
            idx_next = torch.multinomial(prob, num_samples=1) # (B, 1)
            # append the sampled index into the running sequence
            idx = torch.cat((idx, idx_next), dim= 1) # (B,T+1)
        return idx
# define model 
model = TransformerLanguageModel()
model = model.to(device)
print("model created successfully!")

# create optimizer 
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Evaluate loss 
@torch.no_grad()
def evaluate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# training loop 
print("training started...")
for iter in range(max_iterations):
    #once every once a while evalauate loss on the train adn eval set
    if iter % eval_interval == 0:
        losses = evaluate_loss()
        print(f"step {iter}/ {max_iterations},  train_loss: {losses['train']:.4f}, val_loss: {losses['val']:.4f}")
    # sample from the batch
    xb, yb = get_batch("train")
    # forward pass and evaluate the loss 
    logits, loss = model(xb,yb)
    # backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # update 
    optimizer.step()
print("training complete!")

# generate sample text 

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens= 500)[0].tolist()))
