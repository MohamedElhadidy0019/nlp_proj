import torch
import torch.nn as nn
from transformers import GPT2Model, AutoTokenizer, BertModel  # Changed to AutoTokenizer
from torch.utils.data import DataLoader, Dataset, random_split
import os
from typing import List, Tuple, Dict
from tqdm import tqdm
import numpy as np
from dataset_class import EntityDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau 

TXT_FILE_PATH = '/home/mohamed/repos/nlp_proj/EN/raw-documents'


class EntityClassifier(nn.Module):
    def __init__(self, freeze_base=True):
        super().__init__()
        # Define class mappings
        self.main_classes = ['Antagonist', 'Protagonist', 'Innocent']
        self.subclasses = {
            'Antagonist': ['Instigator', 'Conspirator', 'Tyrant', 'Foreign Adversary', 
                          'Traitor', 'Spy', 'Saboteur', 'Corrupt', 'Incompetent', 
                          'Terrorist', 'Deceiver', 'Bigot'],
            'Protagonist': ['Guardian', 'Martyr', 'Peacemaker', 'Rebel', 'Underdog', 'Virtuous'],
            'Innocent': ['Forgotten', 'Exploited', 'Victim', 'Scapegoat']
        }
        
        # Initialize GPT2
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        # self.gpt2 = BertModel.from_pretrained('bert-base-uncased')
        if freeze_base:
            for param in self.gpt2.parameters():
                param.requires_grad = False
        
        # Classification heads
        hidden_size = 768  # GPT2's hidden size
        
        # # Main classifier with multiple layers
        # self.main_classifier = nn.Sequential(
        #     nn.Linear(hidden_size, 512),
        #     nn.LayerNorm(512),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
            
        #     nn.Linear(512, 256),
        #     nn.LayerNorm(256),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),

        #     nn.Linear(256, 256),
        #     nn.LayerNorm(256),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
            
        #     nn.Linear(256, 128),
        #     nn.LayerNorm(128),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),

        #     nn.Linear(128, 128),
        #     nn.LayerNorm(128),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
            
        #     nn.Linear(128, len(self.main_classes))
        # )
        
        # Main classifier with multiple layers
        self.main_classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, len(self.main_classes))
        )
        
        # Separate classifier for each main class's subclasses
        self.subclass_classifiers = nn.ModuleDict({
            main_class: nn.Sequential(
                nn.Linear(hidden_size, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.1),
                
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.1),
                
                nn.Linear(128, len(subclasses))
            )
            for main_class, subclasses in self.subclasses.items()
        })

    def forward(self, input_ids, attention_mask, entity_start_pos, entity_end_pos):
        # Get GPT2 outputs
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Extract entity representations
        batch_size = hidden_states.size(0)
        entity_reprs = []
        
        for i in range(batch_size):
            # Get the average of token embeddings for the entity span
            start = entity_start_pos[i]
            end = entity_end_pos[i]
            entity_repr = hidden_states[i, start:end+1].mean(dim=0)  # Average pooling over entity tokens
            entity_reprs.append(entity_repr)
        
        # Stack entity representations
        entity_reprs = torch.stack(entity_reprs)  # [batch_size, hidden_size]
        
        # Get predictions
        main_class_logits = self.main_classifier(entity_reprs)
        subclass_logits = {
            main_class: classifier(entity_reprs)
            for main_class, classifier in self.subclass_classifiers.items()
        }
        
        return main_class_logits, subclass_logits
def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    max_len = min(max([item['input_ids'].size(0) for item in batch]), 1024)
    
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
            input_ids = torch.cat([input_ids, torch.ones(padding_len, dtype=torch.long) * 50256])
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
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, val_loader, device, dataset):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                entity_start_pos = batch['entity_start_pos'].to(device)
                entity_end_pos = batch['entity_end_pos'].to(device)

                main_class_logits, _ = model(
                    input_ids, attention_mask, entity_start_pos, entity_end_pos
                )

                main_labels = torch.tensor([
                    dataset.main_classes.index(mc) for mc in batch['main_class']
                ]).to(device)

                # Get predictions by selecting the index with the highest logit value
                main_class_preds = torch.argmax(main_class_logits, dim=1)

                total_correct += (main_class_preds == main_labels).sum().item()
                total_samples += main_labels.size(0)

                # Collect predictions and true labels for confusion matrix
                all_preds.extend(main_class_preds.cpu().numpy())
                all_labels.extend(main_labels.cpu().numpy())

            except Exception as e:
                print(f"Error during evaluation: {str(e)}")
                continue

    # Calculate accuracy
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=dataset.main_classes))

    # # Plot confusion matrix
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
    #             xticklabels=dataset.main_classes, 
    #             yticklabels=dataset.main_classes)
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    # plt.title("Confusion Matrix")
    # plt.show()

    return accuracy


def train_model(data_file: str, epochs: int = 100, batch_size: int = 16, learning_rate: float = 1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Use AutoTokenizer instead of GPT2Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    dataset = EntityDataset(data_file, tokenizer)
    
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

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
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=2) # Learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    max_grad_norm = 1.0


    val_accuracy = evaluate_model(model, val_loader, device, dataset)
    print(f" INIT Validation Accuracy: {val_accuracy:.4f}")
        
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
                
                main_labels = torch.tensor([
                    dataset.main_classes.index(mc) for mc in batch['main_class']
                ]).to(device)
                
                loss = criterion(main_class_logits, main_labels)
                
                if torch.isnan(loss):
                    print("NaN loss detected, skipping batch")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                # print(f"Loss: {loss.item():.4f}")
                
            except Exception as e:
                print(f"Error in batch: {str(e)}")
                continue
        
        avg_loss = train_loss / len(train_loader)
        scheduler.step(avg_loss)  # Update learning rate based on training loss

        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        val_accuracy = evaluate_model(model, val_loader, device, dataset)
        print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_accuracy:.4f}")


def main():
    train_model('/home/mohamed/repos/nlp_proj/output.csv')

if __name__ == '__main__':
    main()
    #Token indices sequence length is longer than the specified maximum sequence length for this model (1158 > 1024). Running this sequence through the model will result in indexing errors