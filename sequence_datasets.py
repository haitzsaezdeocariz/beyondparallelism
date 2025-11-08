#Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class ZeroDataset(Dataset):
    '''
    Generates a random sequence of integers and always uses a zero output sequence as target.
    This is a dataset to test fidelity destruction.
    '''

    def __init__(self, num_samples, seq_length, vocab_size):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.data = torch.randint(low=0, high=vocab_size, size=(num_samples, seq_length))
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data[idx]
        return x, x*0

class CopyDataset(Dataset):
    '''
    Generates a random sequence of integers and uses the same sequence as the target.
    This is a simple dataset for testing the model's ability to copy sequences.
    '''

    def __init__(self, num_samples, seq_length, vocab_size):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.data = torch.randint(low=0, high=vocab_size, size=(num_samples, seq_length))
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data[idx]
        return x, x

class CycleDataset(Dataset):
    '''
    Generates a random sequence of integers and uses a cyclically shifted version of the sequence as the target.
    This is a dataset for testing the model's ability to learn cyclic patterns.
    '''
    def __init__(self, num_samples, seq_length, vocab_size):
        """
        Generates a random sequence (as in CopyDataset) and produces the target
        by cyclically shifting the input by a fixed shift (here, 1).
        """
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.data = torch.randint(low=0, high=vocab_size, size=(num_samples, seq_length))
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data[idx]
        target = torch.roll(x, shifts=1, dims=0)
        return x, target