import torch
import os
import json


import torch
import torch.nn as nn
from collections import Counter
from typing import List, Dict, Tuple
import re
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from ts.torch_handler.base_handler import BaseHandler
# from ts.utils.util import handler
from collections.abc import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model import CustomEmbedding, TitanModelWithCustomEmbedding
from tokenizer import CustomTokenizer



# @handler
class TitanHandler(BaseHandler):
   def __init__(self):
       self.model = None
       self.tokenizer = None
       self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


   # def initialize(self, context):
   #
   #     properties = context.system_properties
   #     model_dir = properties.get('model_dir')


   #     # Load tokenizer
   #     with open(os.path.join(model_dir, 'tokenizer_config.json'), 'r') as f:
   #         tokenizer_config = json.load(f)
   #     self.tokenizer = CustomTokenizer(**tokenizer_config)
   #     self.tokenizer.load_vocab(os.path.join(model_dir, 'vocab.txt'))


   #     # Load model
   #     checkpoint = torch.load(
   #         os.path.join(model_dir, 'model.pt'),
   #         map_location=self.device
   #     )


   #     # Initialize model (you'll need to match the configuration used during training)
   #     self.model = TitanModelWithCustomEmbedding(
   #         embedding_layer=CustomEmbedding(
   #             vocab_size=self.tokenizer.vocab_size_current,
   #             embedding_dim=512  # Make sure this matches your training config
   #         )
   #     )
   #     self.model.load_state_dict(checkpoint['model_state_dict'])
   #     self.model.to(self.device)
   #     self.model.eval()
   # def initialize(self, context):
   #
   #     properties = context.system_properties
   #     model_dir = properties.get('model_dir')


   #     # Load tokenizer
   #     with open(os.path.join(model_dir, 'tokenizer_config.json'), 'r') as f:
   #         tokenizer_config = json.load(f)
   #     self.tokenizer = CustomTokenizer(**tokenizer_config)
   #     self.tokenizer.load_vocab(os.path.join(model_dir, 'vocab.txt'))


   #     # Load model
   #     checkpoint = torch.load(
   #         os.path.join(model_dir, 'model.pt'),
   #         map_location=self.device
   #     )


   #     # Initialize model with the correct vocabulary size from the loaded tokenizer
   #     self.model = TitanModelWithCustomEmbedding(
   #         embedding_layer=CustomEmbedding(
   #             vocab_size=self.tokenizer.vocab_size_current,  # Use tokenizer's vocab size
   #             embedding_dim=512  # Make sure this matches your training config
   #         ),
   #         max_seq_length=1024,  # Add this to match your training config
   #         depth=12           # Add this to match your training config
   #     )
   #     self.model.load_state_dict(checkpoint['model_state_dict'])
   #     self.model.to(self.device)
   #     self.model.eval()
   def initialize(self, context):


       properties = context.system_properties
       model_dir = properties.get('model_dir')


       # Load tokenizer FIRST
       with open(os.path.join(model_dir, 'tokenizer_config.json'), 'r') as f:
           tokenizer_config = json.load(f)
       self.tokenizer = CustomTokenizer(**tokenizer_config)
       self.tokenizer.load_vocab(os.path.join(model_dir, 'vocab.txt'))


       # Print vocabulary size for debugging
       print(f"Loaded tokenizer vocabulary size: {self.tokenizer.vocab_size_current}")


       # Load checkpoint to check saved model configuration
       checkpoint = torch.load(
           os.path.join(model_dir, 'model.pt'),
           map_location=self.device
       )


       # Print model state dict shapes for debugging
       print("Model checkpoint shapes:")
       for name, param in checkpoint['model_state_dict'].items():
           print(f"{name}: {param.shape}")


       # Initialize model with the EXACT same vocabulary size as in checkpoint
       vocab_size = checkpoint['model_state_dict']['embedding.embedding.weight'].shape[0]
       print(f"Initializing model with vocabulary size: {vocab_size}")


       self.model = TitanModelWithCustomEmbedding(
           embedding_layer=CustomEmbedding(
               vocab_size=vocab_size,  # Use vocabulary size from checkpoint
               embedding_dim=512
           ),
           max_seq_length=1024,
           depth=12
       )


       # Load state dict
       self.model.load_state_dict(checkpoint['model_state_dict'])
       self.model.to(self.device)
       self.model.eval()


   def preprocess(self, data):


       text = data[0].get('body').decode('utf-8')


       # Tokenize input
       input_ids = torch.tensor(
           [self.tokenizer.encode(text)],
           dtype=torch.long
       ).to(self.device)


       attention_mask = (input_ids != self.tokenizer.pad_token_id).float()


       return {
           'input_ids': input_ids,
           'attention_mask': attention_mask
       }


   def inference(self, data):


       with torch.no_grad():
           outputs = self.model(
               data['input_ids'],
               data['attention_mask']
           )
       return outputs


   def postprocess(self, inference_output):


       # Get predicted tokens
       predictions = inference_output.argmax(dim=-1)


       # Decode tokens to text
       response = self.tokenizer.decode(predictions[0].tolist())


       return [{'generated_text': response}]
   def handle(self, data, context):
       """
       Invoke by TorchServe for prediction request.
       Do pre-processing of data, prediction using model and postprocessing of prediction output
      
       Args:
           data: Input data for prediction
           context: Initial context contains model server system properties.
       Returns:
           prediction output
       """
       if not self.initialized:
           self.initialize(context)
          
       model_input = self.preprocess(data)
       model_output = self.inference(model_input)
       return self.postprocess(model_output)
