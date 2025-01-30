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
from llama_dataset import EntityDataset

from dotenv import load_dotenv 
TXT_FILE_PATH = '/home/mohamed/repos/nlp_proj/EN/raw-documents'
load_dotenv()
ACCESS_TOKEN = os.getenv("HUGGING_TOKEN")


def create_classifier(hidden_size, num_subclasses):
    return nn.Sequential(
        nn.Linear(hidden_size, 1024),
        nn.LayerNorm(1024),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(1024, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(512, num_subclasses)
    )


class EntityClassifier(nn.Module):
    def __init__(self, freeze_base=True, subclass_only = False):
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
        last_layer = self.llama.layers[-1]
        # print(last_layer)
        for param in last_layer.parameters():
            param.requires_grad = True
        
        
        # Classification heads
        hidden_size = 2048  # Llama-2 3.2B hidden size
        
        self.feature_extractor=nn.Sequential(
                    nn.Linear(hidden_size*2, hidden_size*2),
                    nn.LayerNorm(hidden_size*2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size*2, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size, hidden_size)            
        )

        self.main_classifier = create_classifier(hidden_size, len(self.main_classes))
        self.antagonist_classifier = create_classifier(hidden_size, len(self.subclasses['Antagonist']))
        self.protagonist_classifier = create_classifier(hidden_size, len(self.subclasses['Protagonist']))
        self.innocent_classifier = create_classifier(hidden_size, len(self.subclasses['Innocent']))


    def forward(self, input_ids, attention_mask, entity_start_pos, entity_end_pos):
        # Get Llama outputs
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Extract entity representations
        batch_size = hidden_states.size(0)
        entity_reprs = []
        last_hidden_states = []
        
        for i in range(batch_size):
            start = entity_start_pos[i]
            end = entity_end_pos[i]
            entity_repr = hidden_states[i, start:end+1].mean(dim=0)
            
            entity_reprs.append(entity_repr)
            last_hidden_states.append(hidden_states[i, -1].unsqueeze(0)[0])

        
        entity_reprs = torch.stack(entity_reprs)
        last_hidden_states = torch.stack(last_hidden_states)

        # Concatenate entity and last hidden state
        entity_reprs = torch.cat([entity_reprs, last_hidden_states], dim=1)
        extracted_vector = self.feature_extractor(entity_reprs)
        
        # Get predictions
        main_class_logits = self.main_classifier(extracted_vector)
        antagonist_logits = self.antagonist_classifier(extracted_vector)
        protagonist_logits = self.protagonist_classifier(extracted_vector)
        innocent_logits = self.innocent_classifier(extracted_vector)

        
        return main_class_logits, antagonist_logits, protagonist_logits, innocent_logits
