from modules.annotator import Annotator
from modules.tokenizer import Tokenizer
import json
from tqdm import tqdm

if __name__ == '__main__':
    annotator = Annotator.create_default("char", "all")
    tokenizer = Tokenizer("char", 0, False, False)
    file = json.load(open('./data/bart_enhance/train.json', 'r'))
    res_list = []
    for line in tqdm(file):
        if line['candi']!= line['output']:
            candi_out, _ = annotator(tokenizer([line['input']])[0], tokenizer([line['candi']])[0], 0)
            gold_out, _ = annotator(tokenizer([line['input']])[0], tokenizer([line['output']])[0], 0)
            candi_out = [x for x in candi_out if x.startswith('A')]
            gold_out = [x for x in gold_out if x.startswith('A')]
            operation = []
            
            for item in gold_out:
                item = item.split('|||')
                idx = item[0].split(' ')
                if item[1] != 'noop':
                    operation.append([int(idx[1]),int(idx[2]),item[1],item[2]])

            for item in candi_out:
                item = item.split('|||')
                idx = item[0].split(' ')
                if item[1] != 'noop':
                    conflic = False
                    for cur_item in operation:
                        if cur_item[0]<int(idx[2])<=cur_item[1] or cur_item[0]<=int(idx[1])<cur_item[1]:
                            conflic = True
                            break
                    if not conflic:
                        operation.append([int(idx[1]),int(idx[2]),item[1],item[2]])
            res = [x[0] for x in tokenizer([line['input']])[0]]
            operation = sorted(operation, key=lambda x:x[0], reverse=True)
            for oper in operation:
                if oper[2] == 'R':
                    res = res[:oper[0]] + res[oper[1]:]
                else:
                    res = res[:oper[0]] + oper[3].split() + res[oper[1]:]
            # print(operation)
            res = ''.join(res)
            res_list.append({
                'input': line['input'],
                'candi': res,
                'output': line['output'],
            })
    json.dump(res_list, open('./data/gold_label_merging/train.json', 'w'), indent=2, ensure_ascii=False)




