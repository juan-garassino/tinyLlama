import torch
import textwrap

def get_model_num_params(model: torch.nn.Module):
    return sum([param.numel() for param in model.parameters()])

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)