# Data.py
from transformers import GPT2Tokenizer
import torch

# Class to handle text data loading and tokenization
class TextDataset:
    def __init__(self, text_file=None, max_seq_len=128):
        # text_file: path to a text file (optional)
        # max_seq_len: maximum sequence length for tokenization
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Load GPT-2 tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set padding token to EOS
        self.max_seq_len = max_seq_len  # Store max sequence length
        
        # Load text data from file or use dummy data
        if text_file:
            with open(text_file, 'r', encoding='utf-8') as f:
                self.text = f.read()  # Read text from file
        else:
            # Default dummy text for testing
            self.text = "Hello, how can I assist you today? This is a Epyac AI model."

    def get_dataloader(self, batch_size=4):
        # Convert text to token IDs, disable max_length limit to avoid warning
        tokens = self.tokenizer.encode(self.text, return_tensors="pt", max_length=None)[0]
        
        # Split tokens into sequences of max_seq_len + 1 (for input-target pairs)
        sequences = []
        if len(tokens) <= self.max_seq_len + 1:
            # If text is shorter than required length, use it as a single sequence
            # Pad to max_seq_len + 1 if needed
            padding_length = self.max_seq_len + 1 - len(tokens)
            if padding_length > 0:
                tokens = torch.cat([tokens, torch.full((padding_length,), self.tokenizer.eos_token_id, dtype=torch.long)])
            sequences.append(tokens)
        else:
            # Otherwise, split into multiple sequences
            for i in range(0, len(tokens) - self.max_seq_len, self.max_seq_len):
                seq = tokens[i:i + self.max_seq_len + 1]
                sequences.append(seq)
            # Handle remaining tokens (if any) as a final sequence
            if len(tokens) % self.max_seq_len != 0:
                last_seq = tokens[-(self.max_seq_len + 1):]
                padding_length = self.max_seq_len + 1 - len(last_seq)
                if padding_length > 0:
                    last_seq = torch.cat([last_seq, torch.full((padding_length,), self.tokenizer.eos_token_id, dtype=torch.long)])
                sequences.append(last_seq)
        
        # Ensure sequences is not empty
        if not sequences:
            raise ValueError("No sequences could be generated from the text. Increase text length or reduce max_seq_len.")
        
        # Create a tensor dataset and dataloader
        dataset = torch.stack(sequences)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def vocab_size(self):
        # Return the size of the tokenizer's vocabulary
        return self.tokenizer.vocab_size

# Example usage for testing
if __name__ == "__main__":
    dataset = TextDataset()  # Initialize with dummy data
    dataloader = dataset.get_dataloader()  # Get dataloader
    for batch in dataloader:
        print(batch.shape)  # Print shape of a batch: (batch_size, max_seq_len + 1)
        break