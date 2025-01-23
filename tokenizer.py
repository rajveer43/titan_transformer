from ast import Dict
from collections import Counter
import re
from typing import List


class CustomTokenizer:
   def __init__(
       self,
       vocab_size: int = 50000,
       min_freq: int = 2,
       special_tokens: List[str] = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
   ):
       self.vocab_size = vocab_size
       self.min_freq = min_freq
       self.special_tokens = special_tokens


       # Initialize special token IDs
       self.pad_token_id = 0
       self.unk_token_id = 1
       self.bos_token_id = 2
       self.eos_token_id = 3


       # Initialize vocabularies
       self.token2idx: Dict[str, int] = {token: idx for idx, token in enumerate(special_tokens)}
       self.idx2token: Dict[int, str] = {idx: token for idx, token in enumerate(special_tokens)}
       self.vocab_size_current = len(special_tokens)


       # Regex for tokenization
       self.token_pattern = re.compile(r'\w+|[^\w\s]')


   def train_from_texts(self, texts: List[str]) -> None:
       """Train tokenizer on a list of texts."""
       # Count word frequencies
       word_counts = Counter()


       for text in texts:
           tokens = self._basic_tokenize(text)
           word_counts.update(tokens)


       # Filter by minimum frequency and vocab size
       filtered_tokens = [
           token for token, count in word_counts.most_common()
           if count >= self.min_freq and token not in self.special_tokens
       ]


       # Add tokens to vocabulary up to vocab_size
       remaining_space = self.vocab_size - len(self.special_tokens)
       for token in filtered_tokens[:remaining_space]:
           self.token2idx[token] = self.vocab_size_current
           self.idx2token[self.vocab_size_current] = token
           self.vocab_size_current += 1


   def _basic_tokenize(self, text: str) -> List[str]:
       """Basic tokenization into words and punctuation."""
       return self.token_pattern.findall(text.lower())


   def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
       """Encode text to token ids."""
       tokens = self._basic_tokenize(text)


       ids = []
       if add_special_tokens:
           ids.append(self.bos_token_id)


       for token in tokens:
           ids.append(self.token2idx.get(token, self.unk_token_id))


       if add_special_tokens:
           ids.append(self.eos_token_id)


       return ids


   def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
       """Decode token ids back to text."""
       tokens = []
       for idx in ids:
           token = self.idx2token.get(idx, "<UNK>")
           if skip_special_tokens and token in self.special_tokens:
               continue
           tokens.append(token)
       return " ".join(tokens)


   def save_vocab(self, path: str) -> None:
       """Save vocabulary to file."""
       with open(path, 'w', encoding='utf-8') as f:
           for token, idx in sorted(self.token2idx.items(), key=lambda x: x[1]):
               f.write(f"{token}\t{idx}\n")


   def load_vocab(self, path: str) -> None:
       """Load vocabulary from file."""
       self.token2idx.clear()
       self.idx2token.clear()
       with open(path, 'r', encoding='utf-8') as f:
           for line in f:
               token, idx = line.strip().split('\t')
               idx = int(idx)
               self.token2idx[token] = idx
               self.idx2token[idx] = token
       self.vocab_size_current = len(self.token2idx)