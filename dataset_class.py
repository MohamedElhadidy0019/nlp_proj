from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import os
import torch
TXT_FILE_PATH = '/home/mohamed/repos/nlp_proj/EN/raw-documents'

class EntityDataset(Dataset):
    def __init__(self, data_file: str, tokenizer):
        self.examples = []
        self.tokenizer = tokenizer
        self.main_classes = ['Antagonist', 'Protagonist', 'Innocent']
        
        print("Loading dataset...")
        with open(data_file, 'r') as f:
            lines = f.readlines()
            
        for line in tqdm(lines, desc="Processing files"):
            fields = line.strip().split(',')
            text_file, entity, start_pos, end_pos, main_class, *subclasses = fields
            # assert that text_file is str, entity is str, start_pos is int, end_pos is int, main_class is str
            # and subclasses is a list of strings
            try:
                assert isinstance(text_file, str)
                assert isinstance(entity, str)
                assert isinstance(int(start_pos), int)
                assert isinstance(int(end_pos), int)
                assert isinstance(main_class, str)
                assert isinstance(subclasses, list)
            except AssertionError:
                print(f"Error processing line: {line}")
                continue


            text_file = os.path.join(TXT_FILE_PATH, text_file)
            
            try:
                with open(text_file, 'r') as txt_f:
                    article_text = txt_f.read()

                # Tokenize the entire text
                full_text_tokens = tokenizer.encode(article_text, add_special_tokens=False)
                full_text_len = len(full_text_tokens)

                # Tokenize the entity
                entity_tokens = tokenizer.encode(entity, add_special_tokens=False)
                entity_len = len(entity_tokens)

                # Tokenize text before the entity to determine its token position
                text_before = article_text[:int(start_pos)]
                tokens_before = tokenizer.encode(text_before, add_special_tokens=False)
                token_start = len(tokens_before)
                token_end = token_start + entity_len - 1

                # Case 1: Text length <= 1024 tokens
                if full_text_len <= 1024:
                    inputs = tokenizer(
                        article_text,
                        padding="max_length",
                        truncation=False,
                        max_length=1024
                    )

                    # No adjustment needed for token positions
                    self.examples.append({
                        'input_ids': torch.tensor(inputs['input_ids']),
                        'attention_mask': torch.tensor(inputs['attention_mask']),
                        'entity_start_pos': torch.tensor(token_start),
                        'entity_end_pos': torch.tensor(token_end),
                        'main_class': main_class,
                        'subclasses': subclasses
                    })

                # Case 2: Text length > 1024 tokens
                else:
                    context_window = 1024
                    half_window = context_window // 2

                    # Determine the context window around the entity
                    context_start = max(0, token_start - half_window)
                    context_end = min(full_text_len, token_end + half_window)

                    # Adjust context to always be 1024 tokens
                    if context_end - context_start < context_window:
                        if context_start == 0:  # Expand end if start is at the beginning
                            context_end = context_start + context_window
                        elif context_end == full_text_len:  # Expand start if end is at the end
                            context_start = context_end - context_window

                    # Extract the 1024-token context
                    context_tokens = full_text_tokens[context_start:context_end]

                    # Adjust entity positions within the new context
                    adjusted_token_start = token_start - context_start
                    adjusted_token_end = token_end - context_start

                    # Tokenize the context
                    inputs = tokenizer.decode(context_tokens, skip_special_tokens=True)
                    inputs = tokenizer(
                        inputs,
                        padding="max_length",
                        truncation=True,
                        max_length=1024
                    )

                    # Add the example
                    self.examples.append({
                        'input_ids': torch.tensor(inputs['input_ids']),
                        'attention_mask': torch.tensor(inputs['attention_mask']),
                        'entity_start_pos': torch.tensor(adjusted_token_start),
                        'entity_end_pos': torch.tensor(adjusted_token_end),
                        'main_class': main_class,
                        'subclasses': subclasses
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
