import codecs
import glob
import os
import re

from joblib import Parallel, delayed
from tqdm import tqdm

n_pattern = r"(\n{2,})"
o_pattern = r"( +)|(\r+)|(\t+)"


def write_file(file, out_file):
    with codecs.open(file, 'r', encoding='UTF-8') as f, codecs.open(out_file, 'a', encoding='UTF-8') as out:
        content = f.read()
        content = re.sub(n_pattern, '\n', content)
        content = re.sub(o_pattern, '', content)
        out.write(content)
    return os.path.basename(file)


def get_data(filepath="/home/jiangzheng/data/daizhige/daizhigev20-master"):
    file_names = glob.glob(filepath + "/**/*.txt", recursive=True)
    out_file = os.path.join(filepath, "../out.txt")
    result = Parallel(n_jobs=-1, backend='threading')(delayed(write_file)(file, out_file) for file in tqdm(file_names))
    if None in result:
        raise ValueError


if __name__ == '__main__':
    get_data()
    # with open("/home/jiangzheng/data/daizhige/daizhigev20-master/子藏/算法/简平仪说.txt",'r', encoding='utf-8') as f:
    #     content = f.read()
    #     content = re.sub(pattern, '', content)
    #     print(content)
