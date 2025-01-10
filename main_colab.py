


def main():
    #/content/nlp_proj/split/train.csv
    train_file = '/content/nlp_proj/split/train.csv'
    val_file = '/content/nlp_proj/split/val.csv'
    article_txt_path = '/content/nlp_proj/split/EN+PT_txt_files'
    model_save_path = '/content/drive/MyDrive/nlp_llama/llama_save'
    logs_path = '/content/drive/MyDrive/nlp_llama/llama_logs'
    train_model(train_file, val_file, article_txt_path,model_save_path, logs_path, 10, 2)
if __name__ == '__main__':
    main()



ACCESS_TOKEN='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'
