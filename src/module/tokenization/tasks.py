import tiktoken
import torch

class Tokenizer:
    def __init__(self):
        self.enc = tiktoken.encoding_for_model("gpt-4o")

    def encode(self, text: str):
        return self.enc.encode(text)
    
    def decode(self, token_ids: list[int]) -> str:
        return self.enc.decode(token_ids)
    
def encode(text):
    enc = tiktoken.encoding_for_model("gpt-4o")
    return torch.tensor(enc.encode(text))