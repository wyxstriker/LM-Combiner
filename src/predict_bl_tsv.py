from transformers import (
    BartForConditionalGeneration,
    BertTokenizer,
    DataCollatorForSeq2Seq 
)
import random
import datasets
from torch.utils.data import DataLoader
import torch, os, json
from collections import OrderedDict
    from tqdm import tqdm

model_path = './output/cbart_large'
# model_path = './cbart_large_k/checkpoint-5000'

dev_path = './data/fcgec/FCGEC_valid.csv'
# dev_path = './data/fcgec/FCGEC_train.csv'
predict_path='./data/bart_enhance'

bs = 32
beam = 1

if __name__ == '__main__':
    os.makedirs(predict_path, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)

    model.eval()
    model.to(device)
    dev_db = datasets.load_dataset('csv', header=None, data_files={'train': dev_path})['train']
    column_names = dev_db.column_names

    def preprocess_function(examples):
        inputs = examples['1']
        model_inputs = tokenizer(inputs, max_length=256, padding='do_not_pad', truncation=True)
        return model_inputs
    
    dev_db_new = dev_db.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names
    )

    dl = DataLoader(dev_db_new, batch_size=bs, shuffle=False, collate_fn=DataCollatorForSeq2Seq(tokenizer))

    idx = -1
    with open(os.path.join(predict_path, './dev.json'), 'w') as f:
        res_dict = []
        for batch in tqdm(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            res = model.generate(   batch['input_ids'], 
                                    attention_mask=batch['attention_mask'],
                                    num_beams=beam,
                                    do_sample=False,
                                    max_length=256
                                )
            for item in tokenizer.batch_decode(res, skip_special_tokens=True, clean_up_tokenization_spaces=True):
                idx += 1
                insert_item = {}
                output_text = ''.join(item.split(' ')).strip()
                # insert_item['id'] = dev_db[idx]['0']
                insert_item['input'] = dev_db[idx]['1'].strip()
                insert_item['candi'] = output_text
                insert_item['output'] = dev_db[idx]['4'].split('\t')[0]
                res_dict.append(insert_item)
        json.dump(res_dict, f, ensure_ascii=False, indent=2)