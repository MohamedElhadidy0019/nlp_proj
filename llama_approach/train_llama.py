import torch
import torch.nn as nn
from transformers import LlamaModel, LlamaTokenizer, AutoTokenizer
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset
import random
import os
from typing import List, Tuple, Dict
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from sklearn.metrics import confusion_matrix, classification_report
from llama_dataset import EntityDataset
from llama_model import EntityClassifier
from copy import deepcopy

from dotenv import load_dotenv 
TXT_FILE_PATH = '/home/mohamed/repos/nlp_proj/EN/raw-documents'
load_dotenv()
ACCESS_TOKEN = os.getenv("HUGGING_TOKEN")

def train_model(train_file: str, val_file:str, article_txt_path: str, epochs: int = 100, batch_size: int = 1, learning_rate: float = 5e-5):
    main_classes = ['Antagonist', 'Protagonist', 'Innocent']
    subclasses = {
            'Antagonist': ['Instigator', 'Conspirator', 'Tyrant', 'Foreign Adversary', 
                          'Traitor', 'Spy', 'Saboteur', 'Corrupt', 'Incompetent', 
                          'Terrorist', 'Deceiver', 'Bigot'],
            'Protagonist': ['Guardian', 'Martyr', 'Peacemaker', 'Rebel', 'Underdog', 'Virtuous'],
            'Innocent': ['Forgotten', 'Exploited', 'Victim', 'Scapegoat']
        }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Initialize Llama tokenizer
    tokenizer =  AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=ACCESS_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    
    # train_dataset = EntityDataset(train_file, tokenizer, article_txt_path)
    val_dataset = EntityDataset(val_file, tokenizer, article_txt_path)
    train_dataset = deepcopy(val_dataset)
    val_subset_indices = random.sample(range(len(val_dataset)), int(0.1 * len(val_dataset)))
    val_dataset = Subset(val_dataset, val_subset_indices)
    print(f'len of train = {len(train_dataset)}')
    print(f'len of val = {len(val_dataset)}')

    # shuffle both of them
    
    def collate_fn(batch):
        max_len = 2048  # Fixed maximum length for Llama-3.2 1B
        
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
    max_grad_norm = 1.0



    def compute_loss(main_class_logits, antagonist_logits, protagonist_logits, innocent_logits, batch):
        # Map main class and subclasses to indices
        main_class_to_idx = {cls: i for i, cls in enumerate(main_classes)}
        subclass_to_idx = {
            cls: {subcls: i for i, subcls in enumerate(subclasses)}
            for cls, subclasses in subclasses.items()
        }

        # Prepare ground truth labels
        main_class_labels = torch.tensor([main_class_to_idx[cls] for cls in batch['main_class']], dtype=torch.long)
        batch_size = len(batch['main_class'])
        subclass_labels = torch.zeros((batch_size, max(len(subcls) for subcls in subclasses.values())), dtype=torch.float)

        for i, (main_cls, subs) in enumerate(zip(batch['main_class'], batch['subclasses'])):
            for sub in subs:
                subclass_labels[i, subclass_to_idx[main_cls][sub]] = 1# here we get the gnd truth subclass labels

        # Select subclass logits, we get the labels of the the subclass of the right gnd truth main class
        # so what we do here is
        # the max len of subclasses is 12, so we create a tensor of zeros with the same shape as the subclass_labels
        # then we fill the tensor with the logits of the right subclass
        # and compare it with the subclass_labels of the gnd truth main class
        # so if the main class is misclassified
        # the subclass will be compared with the right subclass as we knot it from gnd truth
        subclass_logits = torch.zeros_like(subclass_labels)
        for i, main_pred in enumerate(main_class_labels):
            if main_pred == 0:  # Antagonist
                subclass_logits[i][:len(antagonist_logits[i])] = antagonist_logits[i]
            elif main_pred == 1:  # Protagonist
                subclass_logits[i][:len(protagonist_logits[i])] = protagonist_logits[i]

            elif main_pred == 2:  # Innocent
                # subclass_logits[i] = innocent_logits[i]
                subclass_logits[i][:len(innocent_logits[i])] = innocent_logits[i]


        # Compute losses
        criterion_main= nn.CrossEntropyLoss()
        criterion_sub = nn.BCEWithLogitsLoss()
        # all logitcs to device
        main_class_logits = main_class_logits.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        subclass_logits = subclass_logits.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        main_class_labels = main_class_labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        subclass_labels = subclass_labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        main_loss = criterion_main(main_class_logits, main_class_labels)
        subclass_loss = criterion_sub(subclass_logits, subclass_labels)

        return main_loss, subclass_loss
    
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
                
                main_class_logits, antagonist_logits, protagonist_logits, innocent_logits = model(
                    input_ids, attention_mask, entity_start_pos, entity_end_pos
                )
                main_loss, subclass_loss = compute_loss(main_class_logits, antagonist_logits, protagonist_logits, innocent_logits, batch)
                loss = main_loss + subclass_loss
                
                
                if torch.isnan(loss):
                    print("NaN loss detected, skipping batch")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'main_loss': f"{main_loss.item():.4f}",
                    'subclass_loss': f"{subclass_loss.item():.4f}"
                })
                
            except Exception as e:
                print(f"Error in batch: {str(e)}")
                raise e
                continue
        
        avg_loss = train_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        # Evaluation phase
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Average Training Loss: {avg_loss:.4f}")
        # main_accuracy, subclass_accuracies = evaluate_model(model, val_loader, device, dataset)
        
        # # Save best model
        # if main_accuracy > best_val_accuracy:
        #     best_val_accuracy = main_accuracy
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'main_accuracy': main_accuracy,
        #         'subclass_accuracies': subclass_accuracies
        #     }, 'best_model.pth')
        #     print("Saved new best model!")

        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        print("-" * 50)



def main():
    data_file = "/home/mohamed/repos/nlp_proj/output.csv"
    train_file = '/home/mohamed/repos/nlp_proj/split/train.csv'
    val_file = '/home/mohamed/repos/nlp_proj/split/val.csv'
    article_txt_path = '/home/mohamed/repos/nlp_proj/split/EN+PT_txt_files'
    train_model(train_file, val_file, article_txt_path)
if __name__ == '__main__':
    main()