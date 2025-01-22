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

def test_model(model: nn.Module, dataloader: DataLoader, device: torch.device):
    """
    Evaluate the model on the provided dataset and return predictions.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): Dataloader containing the evaluation dataset.
        device (torch.device): Device to run the evaluation on.

    Returns:
        list: A list of dictionaries containing the main class and predicted subclasses for each sample.
    """
    model.eval()
    predictions_output = []

    main_classes = ['Antagonist', 'Protagonist', 'Innocent']
    main_class_to_subclasses = {
        'Antagonist': ['Instigator', 'Conspirator', 'Tyrant', 'Foreign Adversary',
                       'Traitor', 'Spy', 'Saboteur', 'Corrupt', 'Incompetent',
                       'Terrorist', 'Deceiver', 'Bigot'],
        'Protagonist': ['Guardian', 'Martyr', 'Peacemaker', 'Rebel', 'Underdog', 'Virtuous'],
        'Innocent': ['Forgotten', 'Exploited', 'Victim', 'Scapegoat']
    }

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        entity_start_pos = batch['entity_start_pos'].to(device)
        entity_end_pos = batch['entity_end_pos'].to(device)

        with torch.no_grad():
            main_class_logits, antagonist_logits, protagonist_logits, innocent_logits = model(
                input_ids, attention_mask, entity_start_pos, entity_end_pos
            )
            main_class_predictions = torch.argmax(main_class_logits, dim=1)

        for i in range(len(main_class_predictions)):
            predicted_main_class = main_classes[main_class_predictions[i].item()]

            # Select corresponding subclass logits
            if predicted_main_class == 'Antagonist':
                subclass_logits = antagonist_logits[i]
            elif predicted_main_class == 'Protagonist':
                subclass_logits = protagonist_logits[i]
            elif predicted_main_class == 'Innocent':
                subclass_logits = innocent_logits[i]

            # Convert logits to probabilities and apply threshold
            subclass_probs = torch.sigmoid(subclass_logits)
            subclass_predictions = (subclass_probs > 0.5).cpu().numpy()

            # Get the predicted subclass names
            predicted_subclasses = [
                subclass
                for j, subclass in enumerate(main_class_to_subclasses[predicted_main_class])
                if subclass_predictions[j]
            ]

            predictions_output.append({
                'main_class': predicted_main_class,
                'subclasses': predicted_subclasses
            })

    return predictions_output


def run_model(test_file: str,  article_txt_path: str, model_path:str,  batch_size: int = 1):
    main_classes = ['Antagonist', 'Protagonist', 'Innocent']
    subclasses = {  # Renamed from 'subclasses' to 'subclass_structure'
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
    
    test_dataset = EntityDataset(test_file, tokenizer, article_txt_path)
    print(f'len of train = {len(test_dataset)}')

    
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

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )

    model = EntityClassifier(freeze_base=True).to(device)
    if os.path.exists(os.path.join(model_path)):
        print(f'Loading best model from {model_path}')
        model.load_state_dict(torch.load(model_path))
    else:
        print('No best model found, return')
        return

    


    output_dict = test_model(model, test_loader, device)
   
    return output_dict
    
def save_predictions_to_file(predictions, file_path):
    """
    Save predictions to a text file in the format:
    MainClass, "['Subclass1', 'Subclass2', ...]"

    Args:
        predictions (list): List of dictionaries containing main_class and subclasses.
        file_path (str): Path to the output text file.
    """
    with open(file_path, 'w') as file:
        for prediction in predictions:
            main_class = prediction['main_class']
            subclasses = prediction['subclasses']
            subclasses_str = f'"{subclasses}"'  # Format subclasses as a string list
            file.write(f"{main_class},{subclasses_str}\n")

def main():

    test_file = '/content/nlp_proj/split/test.csv'
    article_txt_path = '/content/nlp_proj/split/EN+PT_txt_files'
    model_path = '/content/drive/MyDrive/nlp_llama/llama_save/best_model_state_dict.pth'
    txt_output_path = '/content/drive/MyDrive/nlp_llama/og_model.txt'
    #/content/nlp_proj/split/train.csv
    # test_file = '/home/mohamed/repos/nlp_proj/split/test.csv'
    # article_txt_path = '/home/mohamed/repos/nlp_proj/split/EN+PT_txt_files'
    # model_path = '/home/mohamed/repos/nlp_proj/model_weights/best_model_state_dict_subclass.pth'
    # txt_output_path = '/home/mohamed/repos/nlp_proj/model_output/og.txt'
    #----------------------------------------------------------

    output_dcit = run_model(test_file, article_txt_path, model_path,1)
    save_predictions_to_file(output_dcit, txt_output_path)

if __name__ == '__main__':
    main()
