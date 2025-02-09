# Project Repository

This repository contains several modules for data preprocessing, training models, and performing analysis, specifically related to NLP tasks involving named entity recognition (NER) and model training using LLaMA and GPT approaches.

## Directory Structure

### `llama_approach/`
This folder includes scripts for the LLaMA-based approach.

- **`llama_model.py`**  
  Contains the definition of the LLaMA model architecture, including layers and forward passes.

- **`train_llama.py`**  
  Main script for training the LLaMA model, handling data loading, model initialization, and training loop.


- **`train_llama_colab.py`**  
    Just having the paths in colab environment
- **`focal_loss.py`**  
  Implements the focal loss function, which is useful for addressing class imbalance in the dataset during training.

- **`train_llama_subclass.py`**  
  The script that utlises the Focal Loss for finetuning the subclass head only, while freezing the base model and the main class head

- **`test_llama.py`**  
  This script that has the inference code of the llama model and prints the output of each data point in the txt file.

- **`llama_model_entity_w_last_token.py`**  
  Modifies the LLaMA model to include entity recognition features with a focus on the last token in the sequence, which ill be trained and analysed in the future.

- **`llama_dataset.py`**  
  Defines the dataset class used in the LLaMA approach, including methods for loading, preprocessing, and tokenizing text for model training.


- **`metrics.py`**  
  Implements functions for evaluating the performance of the LLaMA model, such as accuracy, precision, recall, and F1 score. It takes the ground truth csv and the inference output csv.




### `gpt_approach/`
This folder contains scripts for the GPT-based approach.
which was used as a proof of concept locally before the LLaMA approach.

- **`train.py`**  
  Main script for training the GPT model. It loads data, initializes the model, and handles the training process.

- **`dataset_class.py`**  
  Defines the dataset class, responsible for loading and preprocessing data into a format suitable for GPT training.



### `data_preprocessing/`
This folder includes Python scripts for various data preprocessing tasks.

- **`txt2csv.py`**  
  Converts raw text data into CSV format, making it easier to manipulate for training machine learning models.

- **`validate_named_entity.py`**  
  Contains functions to validate named entity annotations in the dataset, ensuring that entities are labeled correctly.

- **`contextlen.py`**  
  Computes the context length for each text sample, which helps in segmenting the data or adjusting model input lengths.

- **`concat_2_txt.py`**  
  Concatenates multiple text files into a single file, possibly for easier processing or training.

- **`train_test_val_split.py`**  
  Splits the data into training, testing, and validation sets to ensure proper model evaluation and tuning.
