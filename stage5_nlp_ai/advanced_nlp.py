"""
Stage 5: Advanced NLP and AI
Author: Jajitha
Course: Creating Your First Python Program - UST
Description: Transformers, LLMs and HCI in NLP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import re
from collections import Counter


# =============================================
# 1. TEXT PROCESSING PIPELINE
# =============================================

class TextProcessor:
    """
    Advanced text processing pipeline
    simulating transformer preprocessing.
    
    Author: Jajitha
    """
    
    def __init__(self):
        self.vocab = {}
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': 4
        }
        self.vocab.update(self.special_tokens)
        self.next_id = len(self.special_tokens)
    
    def tokenize(self, text):
        """Basic wordpiece-style tokenization"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s가-힣]', '', text)
        tokens = text.split()
        return tokens
    
    def build_vocab(self, texts):
        """Build vocabulary from texts"""
        all_tokens = []
        for text in texts:
            tokens = self.tokenize(text)
            all_tokens.extend(tokens)
        
        token_counts = Counter(all_tokens)
        
        for token, count in token_counts.most_common():
            if token not in self.vocab and count >= 2:
                self.vocab[token] = self.next_id
                self.next_id += 1
        
        print(f"Vocabulary size: {len(self.vocab)}")
        return self.vocab
    
    def encode(self, text, max_length=128):
        """
        Encode text to token IDs
        simulating BERT encoding
        """
        tokens = ['[CLS]'] + self.tokenize(text) + ['[SEP]']
        
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab['[UNK]'])
        
        # Padding or truncation
        if len(ids) < max_length:
            attention_mask = [1] * len(ids) + [0] * (max_length - len(ids))
            ids = ids + [self.vocab['[PAD]']] * (max_length - len(ids))
        else:
            ids = ids[:max_length]
            attention_mask = [1] * max_length
        
        return {
            'input_ids': ids,
            'attention_mask': attention_mask,
            'token_count': sum(attention_mask)
        }
    
    def decode(self, ids):
        """Decode token IDs back to text"""
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = []
        for id in ids:
            token = id_to_token.get(id, '[UNK]')
