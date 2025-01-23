import os
import json
import torch
from tokenizer import CustomTokenizer

def prepare_model_for_serving(model_path: str, save_dir: str, tokenizer: CustomTokenizer):
    """
    Prepare model artifacts for TorchServe.

    Args:
        model_path (str): Path to the saved model checkpoint.
        save_dir (str): Directory to save the prepared model artifacts.
        tokenizer (CustomTokenizer): An instance of the tokenizer to save configuration and vocabulary.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load the model from the checkpoint
    model = torch.load(model_path, map_location=torch.device('cpu'))

    # Save the model in the specified directory
    torch.save(model, os.path.join(save_dir, 'model.pt'))

    # Prepare tokenizer configuration
    tokenizer_config = {
        'vocab_size': tokenizer.vocab_size,
        'min_freq': tokenizer.min_freq,
        'special_tokens': tokenizer.special_tokens,
    }

    # Save tokenizer configuration as JSON
    with open(os.path.join(save_dir, 'tokenizer_config.json'), 'w') as f:
        json.dump(tokenizer_config, f)

    # Save tokenizer vocabulary
    tokenizer.save_vocab(os.path.join(save_dir, 'vocab.txt'))


# Example usage
# Create an instance of CustomTokenizer
tokenizer = CustomTokenizer(
    vocab_size=50000,
    min_freq=2,
    special_tokens=['<pad>', '<eos>', '<bos>', '<unk>']
)

# Call the function with the correct parameters
prepare_model_for_serving('model/checkpoints/best.pt', 'model_store', tokenizer)
