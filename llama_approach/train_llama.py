import torch
import torch.nn as nn
from transformers import LlamaModel, LlamaTokenizer, AutoTokenizer
from torch.utils.data import DataLoader, Dataset, random_split
import os
from typing import List, Tuple, Dict
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from sklearn.metrics import confusion_matrix, classification_report # test

from dotenv import load_dotenv 
TXT_FILE_PATH = '/home/mohamed/repos/nlp_proj/EN/raw-documents'
load_dotenv()
ACCESS_TOKEN = os.getenv("HUGGING_TOKEN")
class EntityDataset(Dataset):
    def __init__(self, data_file: str, tokenizer):
        self.examples = []
        self.tokenizer = tokenizer
        self.main_classes = ['Antagonist', 'Protagonist', 'Innocent']
        self.max_length = 4096  # Extended context length
        
        print("Loading dataset...")
        with open(data_file, 'r') as f:
            lines = f.readlines()
            
        for line in tqdm(lines, desc="Processing files"):
            fields = line.strip().split(',')
            text_file, entity, start_pos, end_pos, main_class, *subclasses = fields
            
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

class EntityClassifier(nn.Module):
    def __init__(self, freeze_base=True):
        super().__init__()
        self.main_classes = ['Antagonist', 'Protagonist', 'Innocent']
        self.subclasses = {
            'Antagonist': ['Instigator', 'Conspirator', 'Tyrant', 'Foreign Adversary', 
                          'Traitor', 'Spy', 'Saboteur', 'Corrupt', 'Incompetent', 
                          'Terrorist', 'Deceiver', 'Bigot'],
            'Protagonist': ['Guardian', 'Martyr', 'Peacemaker', 'Rebel', 'Underdog', 'Virtuous'],
            'Innocent': ['Forgotten', 'Exploited', 'Victim', 'Scapegoat']
        }
        
        # Initialize Llama-2
        self.llama = LlamaModel.from_pretrained("meta-llama/Llama-3.2-1B", token=ACCESS_TOKEN)
        if freeze_base:
            for param in self.llama.parameters():
                param.requires_grad = False
        
        # Classification heads
        hidden_size = 2048  # Llama-2 3.2B hidden size
        
        self.main_classifier = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, len(self.main_classes))
        )
        
        self.subclass_classifiers = nn.ModuleDict({
            main_class: nn.Sequential(
                nn.Linear(hidden_size, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, len(subclasses))
            )
            for main_class, subclasses in self.subclasses.items()
        })

    def forward(self, input_ids, attention_mask, entity_start_pos, entity_end_pos):
        # Get Llama outputs
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Extract entity representations
        batch_size = hidden_states.size(0)
        entity_reprs = []
        
        for i in range(batch_size):
            start = entity_start_pos[i]
            end = entity_end_pos[i]
            entity_repr = hidden_states[i, start:end+1].mean(dim=0)
            entity_reprs.append(entity_repr)
        
        entity_reprs = torch.stack(entity_reprs)
        
        # Get predictions
        main_class_logits = self.main_classifier(entity_reprs)
        subclass_logits = {
            main_class: classifier(entity_reprs)
            for main_class, classifier in self.subclass_classifiers.items()
        }
        
        return main_class_logits, subclass_logits
def train_model(data_file: str, epochs: int = 100, batch_size: int = 1, learning_rate: float = 5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Initialize Llama tokenizer
    tokenizer =  AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=ACCESS_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = EntityDataset(data_file, tokenizer)
    
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    def collate_fn(batch):
        max_len = 4096  # Fixed maximum length for Llama-2
        
        input_ids_batch = []
        attention_mask_batch = []
        entity_start_pos_batch = []
        entity_end_pos_batch = []
        main_class_batch = []
        subclasses_batch = []
        
        for item in batch:
            input_ids = item['input_ids'][:max_len]
            attention_mask = item['attention_mask'][:max_len]
            entity_start = min(item['entity_start_pos'], max_len-1)
            entity_end = min(item['entity_end_pos'], max_len-1)
            
            padding_len = max_len - input_ids.size(0)
            
            if padding_len > 0:
                input_ids = torch.cat([input_ids, torch.ones(padding_len, dtype=torch.long) * tokenizer.pad_token_id])
                attention_mask = torch.cat([attention_mask, torch.zeros(padding_len, dtype=torch.long)])
                
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            entity_start_pos_batch.append(entity_start)
            entity_end_pos_batch.append(entity_end)
            main_class_batch.append(item['main_class'])
            subclasses_batch.append(item['subclasses'])
        
        return {
            'input_ids': torch.stack(input_ids_batch),
            'attention_mask': torch.stack(attention_mask_batch),
            'entity_start_pos': torch.stack(entity_start_pos_batch),
            'entity_end_pos': torch.stack(entity_end_pos_batch),
            'main_class': main_class_batch,
            'subclasses': subclasses_batch
        }

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    model = EntityClassifier(freeze_base=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    criterion = nn.CrossEntropyLoss()
    max_grad_norm = 1.0

    # Initialize best metrics for model saving
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    
    def compute_subclass_labels(batch_subclasses, main_class, subclass_list):
        """Helper function to create one-hot encoded subclass labels"""
        labels = torch.zeros(len(batch_subclasses), len(subclass_list))
        for i, subclasses in enumerate(batch_subclasses):
            for subclass in subclasses:
                if subclass in subclass_list:
                    labels[i, subclass_list.index(subclass)] = 1
        return labels

    def evaluate_model(model, data_loader, device, dataset):
        model.eval()
        total_loss = 0
        main_correct = 0
        subclass_correct = {cls: 0 for cls in model.main_classes}
        subclass_total = {cls: 0 for cls in model.main_classes}
        all_main_preds = []
        all_main_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                entity_start_pos = batch['entity_start_pos'].to(device)
                entity_end_pos = batch['entity_end_pos'].to(device)
                
                main_class_logits, subclass_logits = model(
                    input_ids, attention_mask, entity_start_pos, entity_end_pos
                )
                
                # Main class evaluation
                main_labels = torch.tensor([
                    dataset.main_classes.index(mc) for mc in batch['main_class']
                ]).to(device)
                
                main_preds = torch.argmax(main_class_logits, dim=1)
                main_correct += (main_preds == main_labels).sum().item()
                
                all_main_preds.extend(main_preds.cpu().numpy())
                all_main_labels.extend(main_labels.cpu().numpy())
                
                # Subclass evaluation for each main class
                for main_class, subclass_list in model.subclasses.items():
                    mask = torch.tensor([mc == main_class for mc in batch['main_class']]).to(device)
                    if mask.sum() > 0:
                        subclass_labels = compute_subclass_labels(
                            [sc for i, sc in enumerate(batch['subclasses']) if batch['main_class'][i] == main_class],
                            main_class,
                            subclass_list
                        ).to(device)
                        
                        subclass_preds = (torch.sigmoid(subclass_logits[main_class][mask]) > 0.5).float()
                        correct = (subclass_preds == subclass_labels[mask]).sum().item()
                        total = mask.sum().item() * len(subclass_list)
                        
                        subclass_correct[main_class] += correct
                        subclass_total[main_class] += total

        # Calculate metrics
        main_accuracy = main_correct / len(data_loader.dataset)
        subclass_accuracies = {
            cls: subclass_correct[cls] / max(subclass_total[cls], 1)
            for cls in model.main_classes
        }
        
        # Print detailed classification report for main classes
        print("\nMain Class Classification Report:")
        print(classification_report(
            all_main_labels, 
            all_main_preds, 
            target_names=dataset.main_classes
        ))
        
        # Print subclass accuracies
        print("\nSubclass Accuracies:")
        for cls, acc in subclass_accuracies.items():
            print(f"{cls}: {acc:.4f}")
            
        return main_accuracy, subclass_accuracies

    # Initial evaluation
    # print("\nInitial Evaluation:")
    # main_accuracy, subclass_accuracies = evaluate_model(model, val_loader, device, dataset)
    # print(f"Initial Main Class Accuracy: {main_accuracy:.4f}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                entity_start_pos = batch['entity_start_pos'].to(device)
                entity_end_pos = batch['entity_end_pos'].to(device)
                
                main_class_logits, subclass_logits = model(
                    input_ids, attention_mask, entity_start_pos, entity_end_pos
                )
                
                # Main class loss
                main_labels = torch.tensor([
                    dataset.main_classes.index(mc) for mc in batch['main_class']
                ]).to(device)
                main_loss = criterion(main_class_logits, main_labels)
                
                # # Subclass losses
                # subclass_loss = 0
                # for main_class, subclass_list in model.subclasses.items():
                #     mask = torch.tensor([mc == main_class for mc in batch['main_class']]).to(device)
                #     if mask.sum() > 0:
                #         subclass_labels = compute_subclass_labels(
                #             [sc for i, sc in enumerate(batch['subclasses']) if batch['main_class'][i] == main_class],
                #             main_class,
                #             subclass_list
                #         ).to(device)
                        
                #         subclass_loss += nn.BCEWithLogitsLoss()(
                #             subclass_logits[main_class][mask],
                #             subclass_labels[mask]
                #         )
                
                # Combined loss
                # loss = main_loss + 0.5 * subclass_loss  # Weight factor for subclass loss
                loss = main_loss
                
                if torch.isnan(loss):
                    print("NaN loss detected, skipping batch")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'main_loss': f"{main_loss.item():.4f}"
                    # 'subclass_loss': f"{subclass_loss.item():.4f}"
                })
                
            except Exception as e:
                print(f"Error in batch: {str(e)}")
                continue
        
        avg_loss = train_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        # Evaluation phase
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Average Training Loss: {avg_loss:.4f}")
        main_accuracy, subclass_accuracies = evaluate_model(model, val_loader, device, dataset)
        
        # Save best model
        if main_accuracy > best_val_accuracy:
            best_val_accuracy = main_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'main_accuracy': main_accuracy,
                'subclass_accuracies': subclass_accuracies
            }, 'best_model.pth')
            print("Saved new best model!")

        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        print("-" * 50)



def main():
    data_file = "/home/mohamed/repos/nlp_proj/output.csv"
    train_model(data_file)
if __name__ == '__main__':
    main()