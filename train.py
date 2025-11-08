#Imports
import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, dataloader, num_epochs, lr, device):
    '''
    Train the model using the provided dataloader, number of epochs, learning rate, and device.
    Args:
        model (nn.Module): The model to be trained.
        dataloader (DataLoader): DataLoader for the training data.
        num_epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
        device (str): Device to run the training on ('cpu' or 'cuda').
    '''
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

def save_checkpoint(model, path):
    '''
    Save the model's state dictionary to a file.
    Args:
        model (nn.Module): The model to be saved.
        path (str): Path to save the model checkpoint.
    '''
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path, device):
    '''
    Load the model's state dictionary from a file.
    Args:
        model (nn.Module): The model to load the state dictionary into.
        path (str): Path to the model checkpoint.
        device (str): Device to load the model on ('cpu' or 'cuda').
    '''
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model