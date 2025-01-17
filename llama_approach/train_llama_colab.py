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
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Get current date and time
now = datetime.now()

# # Format date and time
# current_date = now.strftime("%B %d, %Y")  # Full month name, day, year
# current_time = now.strftime("%I:%M %p")   # 12-hour format with AM/PM

# # Print the date and time in a pretty format
# print(f"Today's date is: {current_date}")
# print(f"Current time is: {current_time}")


from dotenv import load_dotenv 
TXT_FILE_PATH = '/home/mohamed/repos/nlp_proj/EN/raw-documents'
load_dotenv()
ACCESS_TOKEN = os.getenv("HUGGING_TOKEN")

def output_evaluation_results(results: Dict, logs_path:str, txt_logs_path, epoch:int):
    print("Main Class Accuracy:", results['main_class']['accuracy'])
    print("Main Class F1 Score:", results['main_class']['f1_score'])
    print("Main Class Classification Report:")
    print(results['main_class']['classification_report'])

    print("Subclasses Classification Report:")
    print(results['subclasses']['classification_report'])
    with open(txt_logs_path, 'a') as f:
        f.write("-" * 10+'\n')
        f.write("Validation Results\n")
        f.write("Main Class Accuracy: "+str(results['main_class']['accuracy'])+'\n')
        f.write("Main Class F1 Score: "+str(results['main_class']['f1_score'])+'\n')
        f.write("Main Class Classification Report:\n")
        f.write(str(results['main_class']['classification_report'])+'\n')
        f.write("Subclasses Classification Report:\n")
        f.write(str(results['subclasses']['classification_report'])+'\n')
        f.write(str(results['subclasses']['statistics'])+'\n')
        f.write("-" * 10+'\n')

    print("Plotting Confusion Matrix for Main Class")
    plot_confusion_matrix(
        results['main_class']['confusion_matrix'], 
        class_names=['Antagonist', 'Protagonist', 'Innocent'],
        epoch_num=epoch,
        logs_path=logs_path
    )
    pass
def plot_confusion_matrix(cm, class_names,epoch_num:int, logs_path:str, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    # plt.show()
    plt.savefig(os.path.join(logs_path, f'confusion_matrix_{epoch_num}.png'))

def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device):
    """
    Evaluate the model on the provided dataset.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): Dataloader containing the evaluation dataset.
        device (torch.device): Device to run the evaluation on.

    Returns:
        dict: A dictionary containing accuracy, F1 score, confusion matrices, and classification reports.
    """
    model.eval()
    all_preds_main = []
    all_targets_main = []
    all_preds_sub = []
    all_targets_sub = []

    main_classes = ['Antagonist', 'Protagonist', 'Innocent']
    subclasses = {
            'Antagonist': ['Instigator', 'Conspirator', 'Tyrant', 'Foreign Adversary', 
                          'Traitor', 'Spy', 'Saboteur', 'Corrupt', 'Incompetent', 
                          'Terrorist', 'Deceiver', 'Bigot'],
            'Protagonist': ['Guardian', 'Martyr', 'Peacemaker', 'Rebel', 'Underdog', 'Virtuous'],
            'Innocent': ['Forgotten', 'Exploited', 'Victim', 'Scapegoat']
        }
    subclass_indices = {}
    subclass_stats = {}


    for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        entity_start_pos = batch['entity_start_pos'].to(device)
        entity_end_pos = batch['entity_end_pos'].to(device)

        main_class_targets = [main_classes.index(cls) for cls in batch['main_class']]
        main_class_targets = torch.tensor(main_class_targets).to(device)

        with torch.no_grad():
            main_class_logits, antagonist_logits, protagonist_logits, innocent_logits = model(
                input_ids, attention_mask, entity_start_pos, entity_end_pos
            )
            predictions = torch.argmax(main_class_logits, dim=1)

        all_preds_main.extend(predictions.cpu().tolist())
        all_targets_main.extend(main_class_targets.cpu().tolist())

        # Process subclasses
        for i, subclasses in enumerate(batch['subclasses']):
            for subclass in subclasses:
                if subclass not in subclass_indices:
                    subclass_indices[subclass] = len(subclass_indices)

                all_preds_sub.append(predictions[i].item())
                all_targets_sub.append(subclass_indices[subclass])
         # Process subclasses
        for main_class, sub_classes in subclasses.items():
            for sub_class in sub_classes:
                subclass_stats[sub_class] = {'occurrences': 0, 'correct_predictions': 0}

        for i, subclasses_list in enumerate(batch['subclasses']):
            predicted_main_class = main_classes[predictions[i].item()]
            actual_main_class = main_classes[main_class_targets[i].item()]
            for subclass in subclasses_list:
                subclass_stats[subclass]['occurrences'] += 1
                if predicted_main_class == actual_main_class and subclass in subclasses[predicted_main_class]:
                    subclass_stats[subclass]['correct_predictions'] += 1

        


    # Calculate metrics for main class
    main_class_acc = np.mean(np.array(all_preds_main) == np.array(all_targets_main))
    main_class_f1 = classification_report(all_targets_main, all_preds_main, target_names=main_classes, output_dict=True)['weighted avg']['f1-score']
    main_class_conf_matrix = confusion_matrix(all_targets_main, all_preds_main)

    # Calculate metrics for subclasses
    subclass_conf_matrix = confusion_matrix(all_targets_sub, all_preds_sub, labels=list(subclass_indices.values()))
    subclass_classification_report = classification_report(
        all_targets_sub,
        all_preds_sub,
        labels=list(subclass_indices.values()),
        target_names=list(subclass_indices.keys()),
        output_dict=True
    )

    return {
        'main_class': {
            'accuracy': main_class_acc,
            'f1_score': main_class_f1,
            'confusion_matrix': main_class_conf_matrix,
            'classification_report': classification_report(all_targets_main, all_preds_main, target_names=main_classes)
        },
        'subclasses': {
            'confusion_matrix': subclass_conf_matrix,
            'classification_report': subclass_classification_report,
            'statistics': subclass_stats

        }
    }

def train_model(train_file: str, val_file:str, article_txt_path: str, model_save_path:str, logs_path:str,txt_logs_path:str,
                epochs: int = 100, batch_size: int = 1, learning_rate: float = 5e-5):
    main_classes = ['Antagonist', 'Protagonist', 'Innocent']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Initialize Llama tokenizer
    tokenizer =  AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=ACCESS_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = EntityDataset(train_file, tokenizer, article_txt_path)
    val_dataset = EntityDataset(val_file, tokenizer, article_txt_path)
    # train_dataset = deepcopy(val_dataset)
    # val_subset_indices = random.sample(range(len(val_dataset)), int(0.1 * len(val_dataset)))
    # val_dataset = Subset(val_dataset, val_subset_indices)
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
    if os.path.exists(os.path.join(model_save_path, 'best_model_state_dict.pth')):
        print(f'Loading best model from {os.path.join(model_save_path, "best_model_state_dict.pth")}')
        model.load_state_dict(torch.load(os.path.join(model_save_path, 'best_model_state_dict.pth')))
    else:
        print('No best model found, training from scratch')

    
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


    now = datetime.now() # Format date and time 
    current_date = now.strftime("%B %d, %Y") # Full month name, day, year 
    current_time = now.strftime("%I:%M %p") # 12-hour format with AM/PM # Print the date and time in a pretty format 
    with open(txt_logs_path, 'a') as f:
        f.write(f"Today's date is: {current_date}\n")
        f.write(f"Current time is: {current_time}\n")

    results_before = evaluate_model(model, train_loader, device)
    output_evaluation_results(results_before, logs_path, txt_logs_path, 0)
    return
    
    
    results_before = evaluate_model(model, val_loader, device)
    output_evaluation_results(results_before, logs_path,txt_logs_path, 0)
    best_val_accuracy = results_before['main_class']['accuracy']
    return
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
        with open(txt_logs_path, 'a') as f:
            f.write(f"\nEpoch {epoch+1}/{epochs}\n")
            f.write(f"Average Training Loss: {avg_loss:.4f}\n")

        # main_accuracy, subclass_accuracies = evaluate_model(model, val_loader, device, dataset)
        if epoch %3 == 0:
            results = evaluate_model(model, val_loader, device)
            output_evaluation_results(results, logs_path,txt_logs_path, epoch)
            main_accuracy = results['main_class']['accuracy']
            # # Save best model
            if main_accuracy > best_val_accuracy:
                best_val_accuracy = main_accuracy
                save_dir = os.path.join(model_save_path, 'best_model_state_dict.pth')
                torch.save(model.state_dict(), save_dir)
            print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
            print("-" * 50)
            with open(txt_logs_path, 'a') as f:
                f.write(f"Current learning rate: {optimizer.param_groups[0]['lr']}\n")
                f.write("-" * 50+'\n')



def main():
    #/content/nlp_proj/split/train.csv
    train_file = '/content/nlp_proj/split/train.csv'
    val_file = '/content/nlp_proj/split/val.csv'
    article_txt_path = '/content/nlp_proj/split/EN+PT_txt_files'
    model_save_path = '/content/drive/MyDrive/nlp_llama/llama_save'
    logs_path = '/content/drive/MyDrive/nlp_llama/llama_logs'
    txt_logs_path = '/content/drive/MyDrive/nlp_llama/llama_logs/logs.txt'

    #----------------------------------------------------------
    # train_file = '/home/mohamed/repos/nlp_proj/split/train.csv'
    # val_file = '/home/mohamed/repos/nlp_proj/split/val.csv'
    # article_txt_path = '/home/mohamed/repos/nlp_proj/split/EN+PT_txt_files'
    # model_save_path = '/home/mohamed/repos/nlp_proj/split/EN+PT_txt_files'
    # logs_path = '/home/mohamed/repos/nlp_proj/llama_logs'
    # txt_logs_path = '/home/mohamed/repos/nlp_proj/llama_logs/logs.txt'

    train_model(train_file, val_file, article_txt_path,model_save_path, logs_path, txt_logs_path, 10, 8)
if __name__ == '__main__':
    main()
