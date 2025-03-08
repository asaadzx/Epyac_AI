# Train.py
import torch
import torch.nn as nn
from Model import SimpleTransformer
from Data import TextDataset
from datasets import load_dataset

# Use GPU if available
if torch.cuda.is_available():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using cuda on GPU: " + torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

# Function to train the model
def train_model(model, dataloader, epochs=10, lr=0.001, device="cuda"):
    # model: the transformer model to train
    # dataloader: provides batches of data
    # epochs: number of training iterations
    # lr: learning rate
    # device: "cpu" or "cuda" for training
    model = model.to(device)  # Move model to specified device
    criterion = nn.CrossEntropyLoss()  # Loss function for classification
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam optimizer

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)  # Move batch to device
            optimizer.zero_grad()  # Clear gradients

            # Split batch into input (all but last token) and target (all but first token)
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]

            output = model(input_ids)  # Forward pass
            # Use reshape instead of view to handle non-contiguous tensors
            loss = criterion(output.reshape(-1, model.fc_out.out_features), target_ids.reshape(-1))

            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            total_loss += loss.item()  # Accumulate loss

        avg_loss = total_loss / len(dataloader)  # Average loss per epoch
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), "Model.pth")
    print("Model saved to 'Model.pth'")

if __name__ == "__main__":
    # Load WikiText-2 dataset and save as a text file
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    text = " ".join(dataset["train"]["text"])
    with open("wikitext_train.txt", "w") as f:
        f.write(text)
    
    # Initialize dataset with the saved text file
    dataset = TextDataset("wikitext_train.txt")
    dataloader = dataset.get_dataloader(batch_size=4)  # Create dataloader
    
    # Initialize the model with the tokenizer's vocab size
    model = SimpleTransformer(vocab_size=dataset.vocab_size())
    
    # Function to count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    parameter_result = count_parameters(model)
    print(f"Total parameters: {parameter_result}")

    # Train the model
    train_model(model, dataloader, epochs=10, device=device)