import argparse
import json
import os
import sys
import time
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import transformers
from pypinyin import lazy_pinyin, Style
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, BertForSequenceClassification


from datasets.tnews import TnewsDataSet
from log import Logger
from utils import set_random_seed

transformers.logging.set_verbosity_warning()
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Runner:

    def __init__(self, args):
        # config

        self.max_length = args.maxlen
        self.num_workers = args.num_workers
        self.learning_rate = args.lr
        self.warmup_ratio = args.warmup_ratio
        self.epoches = args.epochs
        self.batch_size = args.batch_size
        self.pretrained_model = args.pretrained_model
        self.seed = args.seed
        self.train_print_freq = args.train_print_freq
        self.val_freq = args.val_freq
        self.datasets = args.datasets
        self.train_filename = args.train_filename
        self.dev_filename = args.dev_filename
        self.test_filename = args.test_filename
        self.model_name = args.model_name
        self.config = vars(args)

        # log
        t = time.strftime("%Y_%m_%d_%H点%M分%S秒", time.localtime())
        self.dir_path = f"./trained_models/{self.datasets}/{self.model_name}"
        self.save_path = f"{self.dir_path}/model_{t}.pt"
        self.log_path = f"{self.dir_path}/logs/"
        self.tb_log_path = f"{self.dir_path}/tb/{t}/"
        self.output_filename = f"{self.dir_path}/predict_{t}.json"
        self.config_path = f"{self.dir_path}/config_{t}.json"
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        # 日志文件名按照程序运行时间设置
        log_file_name = self.log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
        # 记录正常的 print 信息
        sys.stdout = Logger(log_file_name)
        # 记录 traceback 异常信息
        sys.stderr = Logger(log_file_name)
        self.writer = SummaryWriter(log_dir=self.tb_log_path)
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        set_random_seed(args.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        self.config.update({
            'model_path': self.save_path,
            'log_path': log_file_name,
            'tb_log_path': self.tb_log_path,
            'output_filename': self.output_filename,
        })
        with open("./alphabet.txt", 'r', encoding='utf-8') as fp:
            self.alphabet_list = ['[UNK]'] + [alpha.strip("\n\r") for alpha in fp.readlines()]
            self.alphabet_dict = {alpha: idx for idx, alpha in enumerate(self.alphabet_list)}
        pprint(self.config)

    def valid(self, model, loader, loss_fn):
        model.eval()
        total_loss = 0.0
        preds = None
        trues = None
        for i, batch in enumerate(loader):
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                labels = inputs['labels']
                outputs = model(**inputs)
                if self.model_name == 'bert-cls':
                    loss = outputs[0]
                    outputs = outputs[1]
                else:
                    loss = loss_fn(outputs, labels)
                total_loss += loss.mean().item()
                if preds is None:
                    preds = outputs.detach().cpu().numpy()
                    trues = labels.detach().cpu().numpy()
                else:
                    preds = np.append(preds, outputs.detach().cpu().numpy(), axis=0)
                    trues = np.append(trues, labels.detach().cpu().numpy(), axis=0)
        val_acc = accuracy_score(preds.argmax(1), trues)
        val_loss = total_loss / len(loader)
        return val_loss, val_acc

    def collate_fn(self, batch):
        data, labels = zip(*batch)
        batch_feats = self.tokenizer.batch_encode_plus(list(data),
                                                       add_special_tokens=True,
                                                       padding=True,
                                                       truncation=True,
                                                       max_length=self.max_length,
                                                       )
        input_ids, attention_mask = torch.tensor(batch_feats['input_ids']), torch.tensor(batch_feats['attention_mask'])
        labels = torch.tensor(labels)
        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels}

    def my_collate_fn(self, batch):
        data, labels = zip(*batch)
        batch_input_ids, batch_attention_mask, batch_pinyin_ids = [], [], []
        for text in data:
            tokens = ["[CLS]"] + self.tokenizer.tokenize(text[:self.max_length-2]) + ["[SEP]"]
            pinyin_list = []
            for char in tokens:
                pinyin = lazy_pinyin(char, style=Style.TONE3, errors=lambda item: '[UNK]')
                alpha_token = [0] * 8
                if pinyin[0] != '[UNK]':
                    for idx, alpha in enumerate(pinyin):
                        alpha_token[idx] = self.alphabet_dict.get(alpha, 0)
                pinyin_list.append(alpha_token)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)

            batch_input_ids.append(torch.tensor(input_ids))
            batch_attention_mask.append(torch.tensor(attention_mask))
            batch_pinyin_ids.append(torch.tensor(pinyin_list))
        input_ids = pad_sequence(batch_input_ids, batch_first=True)
        attention_mask = pad_sequence(batch_attention_mask, batch_first=True)
        pinyin_ids = pad_sequence(batch_pinyin_ids, batch_first=True)
        labels = torch.tensor(labels)
        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'pinyin_ids': pinyin_ids,
                'labels': labels}

    def predict(self, model_path=None):
        model_path = self.save_path if model_path is None else os.path.join(self.dir_path, model_path)
        test_dataset = TnewsDataSet(self.test_filename)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                 num_workers=self.num_workers,
                                 collate_fn=self.collate_fn)
        model = self.get_model(num_labels=len(test_dataset.labels))
        model.to(self.device)
        print(f"Loading: {model_path}")
        model.load_state_dict(torch.load(model_path), strict=True)

        preds = []
        for i, batch in enumerate(test_loader):
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                if self.model_name == 'bert-cls':
                    outputs = outputs[1]
                preds.extend(outputs.argmax(1).tolist())

        print(f'Testing...{self.output_filename}')
        fw = open(self.output_filename, 'w')
        with open(self.test_filename) as fr:
            for line, r in zip(fr, preds):
                line = json.loads(line)
                line = json.dumps({'id': str(line['id']), 'label': preds[r]})
                fw.write(line + '\n')
        fw.close()

    def get_model(self, *args, **kwargs):
        # model
        if self.model_name == 'abert':
            from model.abert import ABERT
            model = ABERT(pretrained=self.pretrained_model, num_labels=kwargs['num_labels'])
        elif self.model_name == 'bert-cls':
            model = BertForSequenceClassification.from_pretrained(self.pretrained_model,
                                                                  num_labels=kwargs['num_labels'])
        else:
            from model.bert import BERTClassifier
            model = BERTClassifier(pretrained=self.pretrained_model, num_labels=kwargs['num_labels'])
        return model

    def run(self):

        # dataset
        train_dataset = TnewsDataSet(self.train_filename)
        dev_dataset = TnewsDataSet(self.dev_filename)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers,
                                  collate_fn=self.my_collate_fn if self.model_name == "abert" else self.collate_fn)
        val_loader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                                collate_fn=self.my_collate_fn if self.model_name == "abert" else self.collate_fn)

        model = self.get_model(num_labels=len(train_dataset.labels))
        model.to(self.device)
        # optim
        no_decay = ['bias', 'LayerNorm.weight']
        weight_decay = 0.01
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optim = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optim,
                                                    num_warmup_steps=self.warmup_ratio * len(
                                                        train_loader) * self.epoches,
                                                    num_training_steps=len(train_loader) * self.epoches)
        loss_fn = nn.CrossEntropyLoss()

        # train
        total_batch = 0
        best_val_acc = -float('inf')
        model.zero_grad()
        for epoch in range(self.epoches):
            total_loss = 0
            for i, batch in enumerate(train_loader):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                labels = inputs['labels']
                outputs = model(**inputs)
                if self.model_name == 'bert-cls':
                    loss = outputs[0]
                    outputs = outputs[1]
                else:
                    loss = loss_fn(outputs, labels)
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                scheduler.step()
                model.zero_grad()
                total_batch += 1
                if total_batch % self.train_print_freq == 0:
                    # training log
                    train_acc = accuracy_score(outputs.argmax(1).detach().cpu().numpy(), labels.detach().cpu().numpy())
                    self.writer.add_scalar("loss/train", total_loss / total_batch, total_batch)
                    self.writer.add_scalar("acc/train", train_acc, total_batch)
                    self.writer.add_scalar("lr/train", scheduler.get_last_lr()[0], total_batch)

                    print(
                        f'Training[{self.datasets}] Epoch[{epoch}][{total_batch % len(train_loader)}/[{len(train_loader)}]]： Lr:{scheduler.get_last_lr()[0]} Train Loss: {loss.item()} Train Acc:{train_acc} ')

                if total_batch % self.val_freq == 0 or i == len(train_loader) - 1:
                    val_loss, val_acc = self.valid(model, val_loader, loss_fn)
                    # save model

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        torch.save(model.state_dict(), self.save_path)
                        if not os.path.exists(self.config_path):
                            with open(self.config_path, 'w', encoding='utf-8') as fp:
                                json.dump(self.config, fp, ensure_ascii=False)
                    print(
                        f'Validating[{self.datasets}] Epoch[{epoch}][{total_batch % len(train_loader)}/[{len(train_loader)}]]： Val Loss:{val_loss} Val Acc:{val_acc} Best Val Acc:{best_val_acc}')

                    # log
                    self.writer.add_scalar("loss/val", val_loss, total_batch)
                    self.writer.add_scalar("acc/val", val_acc, total_batch)
                    model.train()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument('-d', '--datasets', default='tnews', type=str)
    parser.add_argument('-m', '--model_name', default='bert', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--train_print_freq', type=int, default=100)
    parser.add_argument('--val_freq', type=int, default=1000)
    parser.add_argument('--pretrained_model', type=str, default="bert-base-chinese")
    parser.add_argument('--train_filename', type=str, default=f"data/tnews/train.json")
    parser.add_argument('--dev_filename', type=str, default=f"data/tnews/dev.json")
    parser.add_argument('--test_filename', type=str, default=f"data/tnews/test.json")
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--maxlen', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)

    args = parser.parse_args()
    runner = Runner(args)
    runner.run()
    # runner.predict()
