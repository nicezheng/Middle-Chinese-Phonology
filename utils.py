import json
import os.path
import random

import numpy as np
import torch
import zhconv
from pypinyin import Style, lazy_pinyin
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_acc(output, label):
    output = output.argmax(1).cpu().numpy()
    acc = accuracy_score(output, label.cpu().numpy())
    return acc


def word2pinyin(filename):
    data = []
    with open(filename) as f:
        for i, l in tqdm(enumerate(f), desc=f"Loading {filename}: "):
            l = json.loads(l)
            text, label = l['sentence'], l.get('label', '100')
            pinyin = lazy_pinyin(text, style=Style.TONE3, errors='ignore')
            if len(pinyin) > 0:
                data.append(" ".join(pinyin))
    # dirname = os.path.dirname(filename)
    basename = os.path.basename(filename)
    newname = "new_" + basename.split(".")[0]
    # newname = os.path.join(dirname, newname)
    with open(newname, 'w', encoding='utf-8') as fp:
        for item in data:
            fp.write(item.strip("\r\n "))
            fp.write("\n")


def word2zch():
    with open("word2zch.json",'r',encoding='utf-8') as f:
        data = json.loads(f.read())
        print(len(data))
        # 繁体转简体
        data = {k:v for k,v in data.items() if k == zhconv.convert(k, 'zh-hans')}
        print(len(data))
        multipron = {}
        for k,v in data.items():
            if len(v.split('|')) > 1:
                multipron[k]=v
        print(len(multipron))
if __name__ == '__main__':
    word2zch()
