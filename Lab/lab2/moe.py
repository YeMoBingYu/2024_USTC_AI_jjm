import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List
import tiktoken
import os
import hashlib
import inspect
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Tokenizer:
    def __init__(self, dataPath: str):
        with open(dataPath, "r", encoding="utf-8") as f:
            self.dataset = f.read()
        self.generate_vocabulary()

    def generate_vocabulary(self):
        self.char2index = {}
        self.index2char = {}
        
        # Adding special tokens
        self.char2index["<START>"] = 0
        self.char2index["<END>"] = 1
        self.index2char[0] = "<START>"
        self.index2char[1] = "<END>"
        
        # Populate the char2index and index2char with unique characters
        index = 2  # Start from 2 because 0 and 1 are for special tokens
        for char in sorted(set(self.dataset)):
            if char not in self.char2index:
                self.char2index[char] = index
                self.index2char[index] = char
                index += 1

    def encode(self, sentence: str) -> torch.Tensor:
        tokens = [self.char2index["<START>"]]  # Add the start token
        for char in sentence:
            tokens.append(self.char2index.get(char, self.char2index["<END>"]))  # Fallback to <END> for unknown chars
        tokens.append(self.char2index["<END>"])  # Add the end token
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, tokens) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()

        characters = []
        for token in tokens:
            char = self.index2char.get(token, "<UNK>")
            if char not in ["<START>", "<END>"]:
                characters.append(char)
        return "".join(characters)


tokenizer = Tokenizer("input.txt")
sentence = "HELLO WORLD"
encoded = tokenizer.encode(sentence)
print("Encoded:", encoded)
decoded = tokenizer.decode(encoded)
print("Decoded:", decoded)


class ShakespeareDataset(Dataset):
    def __init__(self, filepath, tokenizer, chunk_size):
        self.tokenizer = tokenizer
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
        self.encoded = self.tokenizer.encode(text)
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.encoded) - self.chunk_size

    def __getitem__(self, idx):
        #TODO: 提取一段文本(长度为 chunk_size）作为输入，以及这段文本的每一个字符的下一个字符作为标签
        # example(not correspond to real text): chunk = tensor([ 0, 20, 49, 58, 59])
        #         label = tensor([20, 49, 58, 59, 19])
        # decoded chunk: "The "
        # decoded label: "he T"
        chunk = self.encoded[idx: idx + self.chunk_size]
        label = self.encoded[idx + 1: idx + self.chunk_size + 1]
        return chunk, label


def create_dataloader(filepath, tokenizer, chunk_size, batch_size, shuffle=True):
    dataset = ShakespeareDataset(filepath, tokenizer, chunk_size)
    train_dataset,val_dataset = torch.utils.data.random_split(dataset,[int(len(dataset)*0.8),len(dataset)-int(len(dataset)*0.8)])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader, val_dataloader

# Assuming 'input.txt' exists and is in the same directory as this script
tokenizer = Tokenizer(dataPath="input.txt")
train_dataloader,val_dataloader = create_dataloader('input.txt', tokenizer, chunk_size=200, batch_size=2)

# Test the dataloader
# for i, (inputs, targets) in enumerate(train_dataloader):
#     print(f"Batch {i + 1}")
#     print("Inputs:", inputs)
#     print("Targets:", targets)
#     print("Decoded Inputs:", [tokenizer.decode(input) for input in inputs])
#     print("Decoded Targets:", [tokenizer.decode(target) for target in targets])
#     break  # Just show one batch for demonstration

class HeadAttention(nn.Module):
    def __init__(self, seq_len: int, embed_size: int, hidden_size: int):
        super().__init__()
        # a triangular bool matrix for mask
        self.register_buffer("tril", torch.tril(torch.ones(seq_len, seq_len)))

        # Initialize weight matrices for queries, keys, and values
        self.to_q = nn.Linear(embed_size, hidden_size, bias=False)
        self.to_k = nn.Linear(embed_size, hidden_size, bias=False)
        self.to_v = nn.Linear(embed_size, hidden_size, bias=False)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, embed_size)
        batch_size, seq_len, embed_size = inputs.shape

        # Compute queries, keys, and values
        queries = self.to_q(inputs)  # (batch_size, seq_len, hidden_size)
        keys = self.to_k(inputs)  # (batch_size, seq_len, hidden_size)
        values = self.to_v(inputs)  # (batch_size, seq_len, hidden_size)

        # Scale queries
        d_k = queries.shape[-1]
        queries = queries / (d_k ** 0.5)

        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))  # (batch_size, seq_len, seq_len)

        # Apply the mask
        mask = self.tril[:seq_len, :seq_len]
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len, seq_len)

        # Compute the output
        attention_output = torch.matmul(attention_weights, values)  # (batch_size, seq_len, hidden_size)

        return attention_output  # (batch_size, seq_len, hidden_size)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, head_size: int, seq_len: int, embed_size: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = embed_size // n_heads  # Compute head size

        # Ensure head size divides evenly into embed size
        assert embed_size % n_heads == 0, "embed_size must be divisible by n_heads."

        # Create multiple HeadAttention layers
        self.heads = nn.ModuleList([
            HeadAttention(seq_len, embed_size, self.head_size) for _ in range(n_heads)
        ])

        # Projection layer
        self.projection = nn.Linear(embed_size, embed_size)

    def forward(self, inputs):
        # input: (batch_size, seq_len, embed_size)
        batch_size, seq_len, embed_size = inputs.shape

        # Ensure the embedding size matches the number of heads times the head size
        assert embed_size == self.n_heads * self.head_size, "embed_size must be equal to n_heads * head_size"

        # Compute the output for each head
        head_outputs = [head(inputs) for head in self.heads]  # List of (batch_size, seq_len, head_size)

        # Concatenate all the head outputs
        concatenated_output = torch.cat(head_outputs, dim=-1)  # (batch_size, seq_len, embed_size)

        # Project back to the original embedding size
        output = self.projection(concatenated_output)  # (batch_size, seq_len, embed_size)

        return output  # (batch_size, seq_len, embed_size)


class Expert(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()
        # Initialize two linear layers
        self.fc1 = nn.Linear(embed_size, 4 * embed_size)
        self.fc2 = nn.Linear(4 * embed_size, embed_size)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, embed_size)
        mid = self.fc1(inputs)  # (batch_size, seq_len, 4 * embed_size)
        mid = F.relu(mid)  # Apply non-linear activation function
        outputs = self.fc2(mid)  # (batch_size, seq_len, embed_size)
        return outputs


# batch_size, seq_len, embed_size = 2, 5, 64
# expert = Expert(embed_size)
# inputs = torch.randn(batch_size, seq_len, embed_size)
# outputs = expert(inputs)
# print(outputs.shape)  # Should print: torch.Size([2, 5, 64])

class TopkRouter(nn.Module):
    def __init__(self, embed_size, num_experts, active_experts):
        super().__init__()
        self.num_experts = num_experts
        self.active_experts = active_experts

        # Linear layer to produce logits for experts
        self.router_weights = nn.Linear(embed_size, num_experts)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, embed_size)
        batch_size, seq_len, embed_size = inputs.shape

        # Compute the logits for each expert for each token
        logits = self.router_weights(inputs)  # (batch_size, seq_len, num_experts)

        # Apply Softmax to get normalized weights (alphas)
        router_output = F.softmax(logits, dim=-1)  # (batch_size, seq_len, num_experts)

        # Select the top-k experts with the highest weights for each token
        topk_values, indices = torch.topk(router_output, self.active_experts, dim=-1)  # (batch_size, seq_len, active_experts)

        # Normalize the weights of the selected experts
        topk_normalized = topk_values / topk_values.sum(dim=-1, keepdim=True)  # (batch_size, seq_len, active_experts)

        return topk_normalized, indices

# Example usage

# batch_size, seq_len, embed_size = 2, 5, 64
# num_experts = 10
# active_experts = 3

# router = TopkRouter(embed_size, num_experts, active_experts)
# inputs = torch.randn(batch_size, seq_len, embed_size)
# router_output, indices = router(inputs)

# print("Router output (normalized weights):", router_output)
# print("Selected experts indices:", indices)

class SparseMoE(nn.Module):
    def __init__(self, embed_size: int, num_experts: int, active_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.active_experts = active_experts

        # Initialize the router
        self.router = TopkRouter(embed_size, num_experts, active_experts)

        # Initialize the experts
        self.experts = nn.ModuleList([Expert(embed_size) for _ in range(num_experts)])

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, embed_size)
        batch_size, seq_len, embed_size = inputs.shape

        # Get routing weights and expert indices
        router_output, indices = self.router(inputs)  # (batch_size, seq_len, active_experts)

        # Prepare a tensor for collecting expert outputs
        final_output = torch.zeros_like(inputs)  # (batch_size, seq_len, embed_size)

        # Iterate over the active experts
        for i in range(self.active_experts):
            expert_idx = indices[:, :, i]  # (batch_size, seq_len)

            # Prepare a mask for selecting inputs for the current expert
            mask = torch.zeros(batch_size, seq_len, self.num_experts, device=inputs.device)
            mask.scatter_(2, expert_idx.unsqueeze(-1), 1)

            # Gather inputs for the current expert
            selected_inputs = inputs.unsqueeze(2) * mask.unsqueeze(-1)  # (batch_size, seq_len, num_experts, embed_size)
            selected_inputs = selected_inputs.sum(dim=2)  # (batch_size, seq_len, embed_size)

            # Apply the expert to the selected inputs
            expert_output = self.experts[i](selected_inputs)  # (batch_size, seq_len, embed_size)

            # Accumulate the outputs weighted by the router weights
            final_output += expert_output * router_output[:, :, i].unsqueeze(-1)  # (batch_size, seq_len, embed_size)

        return final_output

# batch_size, seq_len, embed_size = 2, 5, 64
# num_experts = 10
# active_experts = 3

# moe = SparseMoE(embed_size, num_experts, active_experts)
# inputs = torch.randn(batch_size, seq_len, embed_size)
# outputs = moe(inputs)

# print(outputs.shape)  # Should print: torch.Size([2, 5, 64])

class Block(nn.Module):
    def __init__(self, embed_size: int, n_heads: int, seq_len: int, num_experts: int, active_experts: int):
        super().__init__()

        # Multi-Head Attention
        self.attention = MultiHeadAttention(n_heads, embed_size // n_heads, seq_len, embed_size)
        
        # Sparse MoE
        self.sparse_moe = SparseMoE(embed_size, num_experts, active_experts)
        
        # Feed Forward Network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size)
        )

        # Layer Normalization
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.ln3 = nn.LayerNorm(embed_size)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, embed_size)

        # Multi-Head Attention
        attn_output = self.attention(inputs)
        attn_output = self.ln1(inputs + attn_output)  # Add & Norm

        # Sparse MoE
        moe_output = self.sparse_moe(attn_output)
        moe_output = self.ln2(attn_output + moe_output)  # Add & Norm

        # Feed Forward Network
        ff_output = self.feed_forward(moe_output)
        output = self.ln3(moe_output + ff_output)  # Add & Norm

        return output

# Example usage

# batch_size, seq_len, embed_size = 2, 5, 64
# n_heads = 8
# num_experts = 10
# active_experts = 3

# block = Block(embed_size, n_heads, seq_len, num_experts, active_experts)
# inputs = torch.randn(batch_size, seq_len, embed_size)
# outputs = block(inputs)

# print(outputs.shape)  # Should print: torch.Size([2, 5, 64])

class SparseMoETransformer(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, embed_size: int, n_layers: int, n_heads: int, num_experts: int, active_experts: int):
        super().__init__()

        self.seq_len = seq_len
        self.embed_size = embed_size

        # Token Embedding and Positional Encoding
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(seq_len, embed_size)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            Block(embed_size, n_heads, seq_len, num_experts, active_experts) 
            for _ in range(n_layers)
        ])

        # Layer Normalization and Output Linear Layer
        self.layer_norm = nn.LayerNorm(embed_size)
        self.output_layer = nn.Linear(embed_size, vocab_size)

    def forward(self, inputs, labels=None):
        batch_size, seq_len = inputs.shape

        # Create position IDs
        position_ids = torch.arange(seq_len, device=inputs.device).unsqueeze(0).expand(batch_size, -1)

        # Embedding
        token_embeddings = self.token_embedding(inputs)  # (batch_size, seq_len, embed_size)
        position_embeddings = self.position_embedding(position_ids)  # (batch_size, seq_len, embed_size)
        embeddings = token_embeddings + position_embeddings  # (batch_size, seq_len, embed_size)

        # Pass through Transformer Blocks
        x = embeddings
        for block in self.blocks:
            x = block(x)

        # Layer Norm and Linear Layer
        x = self.layer_norm(x)
        logits = self.output_layer(x)  # (batch_size, seq_len, vocab_size)

        # Compute the loss if labels are provided
        if labels is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss

    # def generate(self, inputs, max_new_tokens):
    #     inputs = torch.tensor(inputs).unsqueeze(0)  # Convert to tensor and add batch dimension
    #     device = next(self.parameters()).device
    #     inputs = inputs.to(device)

    #     generated = inputs
    #     for _ in range(max_new_tokens):
    #         # Crop the input to the sequence length limit
    #         if generated.size(1) > self.seq_len:
    #             generated_input = generated[:, -self.seq_len:]
    #         else:
    #             generated_input = generated

    #         logits, _ = self.forward(generated_input)
    #         next_token_logits = logits[:, -1, :]
    #         next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

    #         generated = torch.cat([generated, next_token_id], dim=1)

    #     return generated
    def generate(self, inputs, tokenizer, max_new_tokens):
        encoded_input = tokenizer.encode(inputs)  # Encode input string into tokens
        encoded_input = torch.tensor(encoded_input).unsqueeze(0)  # Convert to tensor and add batch dimension
        device = next(self.parameters()).device  # Get the device where the model is located
        encoded_input = encoded_input.to(device)

        if encoded_input.size(1) > self.seq_len:
            encoded_input = encoded_input[:, :self.seq_len]

        generated = encoded_input
        for _ in range(max_new_tokens):
            if generated.size(1) > self.seq_len:
                generated_input = generated[:, -self.seq_len:]
            else:
                generated_input = generated

            logits, _ = self.forward(generated_input)
            last_logits = logits[:, -1, :]
            next_token_ids = torch.argmax(last_logits, dim=-1)
            next_token_ids = next_token_ids.unsqueeze(-1)
            generated = torch.cat([generated, next_token_ids], dim=1)

        generated_ids = generated[0].tolist()
        generated_text = tokenizer.decode(generated_ids)
        return generated_text


# Example usage
# vocab_size = 10000
# seq_len = 20
# embed_size = 64
# n_layers = 4
# n_heads = 8
# num_experts = 10
# active_experts = 3

# model = SparseMoETransformer(vocab_size, seq_len, embed_size, n_layers, n_heads, num_experts, active_experts)
# inputs = torch.randint(0, vocab_size, (2, seq_len))
# labels = torch.randint(0, vocab_size, (2, seq_len))

# logits, loss = model(inputs, labels)
# print("Logits shape:", logits.shape)  # Should be (batch_size, seq_len, vocab_size)
# print("Loss:", loss.item())

# generated_ids = model.generate([1, 2, 3], max_new_tokens=10)
# print("Generated IDs:", generated_ids)

# Define the train function
def train(model, dataloader, epoch, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    total_loss = 0
    for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch} Loss: {total_loss / len(dataloader)}')
    return total_loss / len(dataloader)

# Define the validate function
def validate(model, dataloader, epoch, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            logits, loss = model(inputs, targets)
            total_loss += loss.item()
    print(f'Epoch {epoch} Validation Loss: {total_loss / len(dataloader)}')
    return total_loss / len(dataloader)

# Create the dataloader
train_dataloader, valid_dataloader = create_dataloader('input.txt', tokenizer, chunk_size=50, batch_size=512)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SparseMoETransformer(
    vocab_size=len(tokenizer.char2index),
    seq_len=50,
    embed_size=64,
    n_layers=3,
    n_heads=8,
    num_experts=8,
    active_experts=2
).to(device)

# Train and validate the model
def run(model, train_dataloader, valid_dataloader, device, epochs=10):
    train_losses = []
    valid_losses = []
    for epoch in range(epochs):
        train_loss = train(model, train_dataloader, epoch, device)
        valid_loss = validate(model, valid_dataloader, epoch, device)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f'Epoch {epoch} Train Loss: {train_loss}, Valid Loss: {valid_loss}')

    # Plot the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), train_losses, label='Train Loss')
    plt.plot(range(epochs), valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()

# Run the training process
run(model, train_dataloader, valid_dataloader, device, epochs=5)

# Save the model
torch.save(model.state_dict(), 'model.pth')

# Load the model
model.load_state_dict(torch.load('model.pth'))

# Generate text
generated_text = model.generate("I could pick my lance", tokenizer=tokenizer,max_new_tokens=300)
#generated_text = model.generate("listen!!!", tokenizer=tokenizer,max_new_tokens=300)
print(generated_text)