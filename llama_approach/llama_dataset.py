import torch
import torch.nn as nn
from transformers import LlamaModel, LlamaTokenizer, AutoTokenizer
from torch.utils.data import DataLoader, Dataset, random_split
import os
from typing import List, Tuple, Dict
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from sklearn.metrics import confusion_matrix, classification_report
from ast import literal_eval
import pandas as pd

class EntityDataset(Dataset):
    def __init__(self, data_file: str, tokenizer, txt_file_path: str):
        self.examples = []
        self.tokenizer = tokenizer
        self.main_classes = ['Antagonist', 'Protagonist', 'Innocent']
        self.max_length = 4096  # Extended context length
        
        print("Loading dataset...")
        with open(data_file, 'r') as f:
            lines = f.readlines()
        df = pd.read_csv(data_file)

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows", unit="row", colour="green"):            #article_id,entity_mention,start_offset,end_offset,main_role,fine_grained_roles

            text_file = row['article_id']
            entity = row['entity_mention']
            start_pos = row['start_offset']
            end_pos = row['end_offset']
            main_class = row['main_role']
            subclasses = literal_eval(row['fine_grained_roles'])
            # if (index %8 == 0):
            #     print(f"Entity: {entity}")
            #     print(f"Start position: {start_pos}")
            #     print(f"End position: {end_pos}")
            #     print(f"Main class: {main_class}")
            #     print(f"Subclasses: {subclasses}")

            try:
                assert isinstance(text_file, str)
                assert isinstance(entity, str)
                assert isinstance(int(start_pos), int)
                assert isinstance(int(end_pos), int)
                assert isinstance(main_class, str)
                assert isinstance(subclasses, list)
            except AssertionError:
                print(f"Error processing row: {row}")
                continue

            text_file = os.path.join(txt_file_path, text_file)
            
            try:
                with open(text_file, 'r') as txt_f:
                    article_text = txt_f.read()

                # Tokenize the entire text
                full_text_tokens = tokenizer.encode(article_text, add_special_tokens=False)
                full_text_len = len(full_text_tokens)

                # Tokenize the entity
                entity_tokens = tokenizer.encode(entity, add_special_tokens=False)
                entity_len = len(entity_tokens)

                # Tokenize text before the entity to find its position
                text_before = article_text[:int(start_pos)]
                tokens_before = tokenizer.encode(text_before, add_special_tokens=False)
                token_start = len(tokens_before)
                token_end = token_start + entity_len - 1

                # Calculate which chunk contains the entity
                chunk_size = self.max_length - 2  # Account for special tokens
                chunk_idx = token_start // chunk_size
                chunk_start = chunk_idx * chunk_size
                chunk_end = min((chunk_idx + 1) * chunk_size, full_text_len)

                # Extract the relevant chunk
                chunk_tokens = full_text_tokens[chunk_start:chunk_end]
                
                # Adjust entity positions within the chunk
                adjusted_token_start = token_start - chunk_start
                adjusted_token_end = token_end - chunk_start

                # Convert to text and retokenize to ensure proper formatting
                chunk_text = tokenizer.decode(chunk_tokens)
                inputs = tokenizer(
                    chunk_text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )

                self.examples.append({
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'entity_start_pos': torch.tensor(adjusted_token_start),
                    'entity_end_pos': torch.tensor(adjusted_token_end),
                    'main_class': main_class,
                    'subclasses': subclasses,
                    'chunk_idx': chunk_idx,
                    'total_chunks': (full_text_len + chunk_size - 1) // chunk_size
                })

            except FileNotFoundError:
                print(f"Warning: Could not find file {text_file}")
            except Exception as e:
                print(f"Error processing {text_file}: {str(e)}")

        print(f"Loaded {len(self.examples)} valid examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
