import json

from pypinyin import Style, lazy_pinyin, pinyin
from torch.utils.data import Dataset
from tqdm import tqdm


class TnewsDataSet(Dataset):
    def __init__(self, filename, tokenizer=None, add_pinyin=False, azh=False):
        super(TnewsDataSet, self).__init__()
        self.max_length = 512
        self.tokenizer = tokenizer
        self.add_pinyin = add_pinyin
        self.azh = azh
        self.azh_dict = {}
        self.labels = [
            '100', '101', '102', '103', '104', '106', '107', '108', '109', '110', '112',
            '113', '114', '115', '116'
        ]

        self.data = []
        with open(filename) as f:
            for i, l in tqdm(enumerate(f), desc=f"Loading {filename}: "):
                l = json.loads(l)
                text, label = l['sentence'], l.get('label', '100')
                self.data.append((text, self.labels.index(label)))
        alphabet_file = "./azh_alpha.txt" if azh else "./alphabet.txt"

        with open(alphabet_file, 'r', encoding='utf-8') as fp:
            self.alphabet_list = [alpha.strip("\n\r") for alpha in fp.readlines()]
            self.alphabet_dict = {alpha: idx for idx, alpha in enumerate(self.alphabet_list)}

        if azh:
            with open("./azh_dict.txt") as fdcit:
                self.azh_dict = json.load(fdcit)

    def __getitem__(self, item):
        sentence, label = self.data[item]
        sentence = sentence[:self.max_length - 2]

        tokenizer_output = self.tokenizer.encode(sentence)
        input_ids = tokenizer_output.ids
        attention_mask = [1] * len(input_ids)
        pinyin_ids = []
        # pinyin
        if self.add_pinyin:
            if self.azh:
                pinyin_ids = self.convert_sentence_to_pinyin_ids(sentence, tokenizer_output, errors_token='[UNK]')
            else:
                pinyin_ids = self.convert_sentence_to_pinyin_ids(sentence, tokenizer_output, errors_token='[UNK]')

        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'pinyin_ids': pinyin_ids,
                'labels': [int(label)]}

    def lazy_azh(self, token):
        azh = self.azh_dict.get(token, '[UNK]')
        if azh == "無":
            return ['[UNK]']
        else:
            return [azh]

    def __len__(self):
        return len(self.data)

    def convert_sentence_to_pinyin_ids(self, sentence, tokenizer_output, errors_token='[UNK]'):
        pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True,errors=lambda x: errors_token)
        pinyin_locs = {}
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            if pinyin_string == errors_token:
                continue
            ids = [0] * 8
            for i, p in enumerate(pinyin_string):
                if p not in self.alphabet_dict.keys():
                    ids = [0] * 8
                    break
                ids[i] = self.alphabet_dict[p]
            # ids = [self.alphabet_dict.get(alpha,0) for alpha in ids]
            pinyin_locs[index] = ids

        pinyin_ids = []
        for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
            if offset[1] - offset[0] != 1:
                pinyin_ids.append([0] * 8)
                continue
            if offset[0] in pinyin_locs:
                pinyin_ids.append(pinyin_locs[offset[0]])
            else:
                pinyin_ids.append([0] * 8)
        return pinyin_ids
