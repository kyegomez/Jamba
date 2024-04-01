import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from jamba.model import Jamba

# Load the dataset from Hugging Face
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
tokenizer.pad_token = tokenizer.eos_token


# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=100,
        return_tensors="pt",
    )


tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type="torch", columns=["input_ids"])


# DataLoader
def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    # Create targets by shifting input_ids one token to the left
    labels = torch.roll(input_ids, -1, dims=-1)
    return input_ids.squeeze(), labels.squeeze()


dataloader = DataLoader(
    tokenized_datasets,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
)

# Initialize the Jamba model with tokenizer's vocab size
model = Jamba(
    dim=512,
    depth=6,
    num_tokens=tokenizer.vocab_size,
    d_state=256,
    d_conv=128,
    heads=8,
    num_experts=8,
    num_experts_per_token=2,
)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        outputs = model(inputs)
        loss = criterion(
            outputs.transpose(1, 2), targets
        )  # Adjust for cross-entropy expecting class dimension at dim=1

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

print("Training complete!")
