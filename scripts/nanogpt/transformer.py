""" Script to implement Transformer based language model"""

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# hyperparameters 
batch_size = 32
block_size = 8
max_iterations = 5000
learning_rate = 1e-3
eval_iters = 200
eval_interval = 500 
device = "cuda" if torch.cuda.is_available() else 'cpu'
n_embd = 32 # no of embedding dimention
n_heads = 4
dropout = 0.2
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

# class implementation for self attention head 

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)) )
        self.dropout = nn.Dropout(dropout) 

    def forward(self,x):
        B, T, C = x.shape
        # k,q,v
        k = self.key(x) #(B, T, head_size)
        q = self.query(x) #(B, T, head_size)
        # compute attention scores ('affinities')
        wei = q @ k.transpose(-2, -1)* self.head_size**-0.5 # (B, T, head_size) @ (B, head_size, T) --> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #(B,T,T)
        wei = F.softmax(wei, dim =-1 ) #(B,T,T)
        wei = self.dropout(wei) 
        # perform weighted aggregation of values
        v = self.value(x) #(B, T, head_size)
        out = wei @ v # (B,T,T) @ (B,T,head_size) ==> (B, T, head_size)
        return out

# class implemetnation fo multihead attention 

class MultiheadAttenion(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # projection layer back into Residual pathway
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x):
        out = torch.cat( [h(x) for h in self.heads], dim = -1) # concatenate in channel dimention
        out = self.dropout(self.proj(out))
        return out

# class implementation for feed forward network 
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), # from the transformer paper
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd), # projection layer back into Residual pathway
            nn.Dropout(dropout), # adding droip outs 
        )
    
    def forward(self, x):
        return self.net(x)
    
# class implementation for tranformer block
class Block(nn.Module):
    """ Transformer block: communication and then computation"""
    def __init__(self,n_embd, n_heads):
        # embedding_dimension and no of heads we like
        super().__init__()
        head_size  = n_embd // n_heads
        self.sa_heads = MultiheadAttenion(n_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # layer norm 1 applied before attention
        self.ln2 = nn.LayerNorm(n_embd) # layer norm 2 applied before ffw 

    def forward(self, x):
        # residual connections--> x = x + computation
        x = x + self.sa_heads(self.ln1(x)) # (B, T, head_size) # here head_size = n_embd
        x = x + self.ffwd(self.ln2(x)) # (B, T, n_embd)
        return x
    
# transformer language model, biagram sampling
class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits of the next token from a look up table
        # nn.Embedding -> A simple lookup table that stores embeddings of a fixed dictionary and size.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.postion_embedding_table = nn.Embedding(block_size, n_embd)
        #self.sa_head = Head(n_embd). --> when single head attention 
        #self.sa_heads = MultiheadAttenion(num_heads, n_embd//num_heads) # 4 heads of 8head size --> when single multihead attention
        #self.ffwd = FeedForward(n_embd, n_embd) # add computation after attention --> when single multihead attention
        self.blocks = nn.Sequential(
            Block(n_embd, n_heads),
            Block(n_embd, n_heads),
            Block(n_embd, n_heads),
            Block(n_embd, n_heads),
            nn.LayerNorm(n_embd), #added right before the computation that project to vocab-size
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx,targets=None):
        B,T = idx.shape
        # idx and targets are of shape (B,T)
        token_embeddings = self.token_embedding_table(idx) # (B,T,n_embd)
        position_embeddings = self.postion_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x = token_embeddings + position_embeddings # (B, T, n_embd) + (B, T, n_embd) -> (B,T,n_embd)
        #x = self.sa_head(x) 
        #x = self.sa_heads(x) # (B, T, head_size) # here head_size = n_embd
        #x = self.ffwd(x) # (B, T, n_embd)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            # pytorch cross entropy expects second dimention as the channel
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is B,T array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the idx to always be block size 
            idx_cond = idx[:, -block_size:]
            # get the predictions 
            logits, loss = self(idx_cond) # (B, T, C) # self here runs the def forward 
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
