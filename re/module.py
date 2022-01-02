import torch
from tqdm import tqdm
import random
import os
import numpy as np

import sys
sys.path.append('/home/sy/project/albert_re/')

from utils.log import logger

from .model import AlbertFC, load_tokenizer, load_config, load_pretrained_model, build_model

from .dataset import GetDataset, get_dataloader, get_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed):
    '''
    set seed
    :param seed:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(2021)

class RE():
    '''
    re
    '''
    def __init__(self, args):
        self.args = args
        self.SPECIAL_TOKEN = args.SPECIAL_TOKEN
        self.label2i = args.LABEL2I
        self.model = None
        self.tokenizer = None

    def train(self):
        self.tokenizer = load_tokenizer(self.args.pretrained_model_path, self.SPECIAL_TOKEN)
        pretrained_model, albertConfig = load_pretrained_model(self.args.pretrained_model_path, self.tokenizer, self.SPECIAL_TOKEN)

        train_set = GetDataset(self.args.train_path, self.tokenizer, self.args.max_length, self.SPECIAL_TOKEN, self.label2i)

        if self.args.dev_path:
            dev_dataset = GetDataset(self.args.dev_path, self.tokenizer, self.args.max_length, self.SPECIAL_TOKEN, self.label2i)
            val_iter = get_dataloader(dev_dataset, batch_size=self.args.batch_size, shuffle=False)
        train_iter = get_dataloader(train_set, batch_size=self.args.batch_size)

        tag_num = len(self.label2i)
        albertfc = AlbertFC(albertConfig, pretrained_model, tag_num)
        self.model = albertfc.to(DEVICE)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=0.1)

        best_val_loss = float('inf')
        for epoch in range(self.args.epochs):
            self.model.train()
            acc_loss = 0
            for item in tqdm(train_iter):
                self.model.zero_grad()
                label = item['label']
                input_ids = item['input_ids']
                attention_mask = item['attention_mask']
                entity_mask = item['entity_mask']

                label = label.to(DEVICE)
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                entity_mask = entity_mask.to(DEVICE)
                out = self.model(input_idx=input_ids, attention_mask=attention_mask, entity_mask=entity_mask)
                item_loss = criterion(out, label)
                acc_loss += item_loss.item()
                item_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                optimizer.step()
            logger.info('epoch: {}, acc_loss: {}'.format(epoch, acc_loss / len(train_iter)))

            if self.args.dev_path:
                val_loss = self.validate(val_iter=val_iter, criterion=criterion)
                # val_loss = self._validate(val_iter=val_iter)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # save model
                    torch.save(self.model.state_dict(), self.args.model_path)
                    # torch.save(self.model, self.args.model_path)
                    logger.info('save model : {}'.format(self.args.model_path))
                logger.info('val_loss: {}, best_val_loss: {}'.format(val_loss, best_val_loss))

            scheduler.step()

    def predict(self, text):
        self.model.eval()
        with torch.no_grad():
            entity_1, entity_2, sentence = text
            if sentence.index(entity_1) > sentence.index(entity_2):  # order
                entity_1, entity_2 = entity_2, entity_1
            sentence = sentence.replace(entity_1, '[E1]' + entity_1 + '[/E1]', 1)
            sentence = sentence.replace(entity_2, '[E2]' + entity_2 + '[/E2]', 1)

            # entity mask: entity_1_start => 1 and entity_2_start => 1 => [0,0,1,0,0,0,1,0]
            sentence = ' '.join(list(sentence))
            sentence = sentence.replace('[ E 1 ]',  '[E1]')
            sentence = sentence.replace('[ / E 1 ]', '[/E1]')
            sentence = sentence.replace('[ E 2 ]', '[E2]')
            sentence = sentence.replace('[ / E 2 ]', '[/E2]')
            sentence = sentence.split()
            entity_mask = [0]  # => [CLS]
            entity_mask.extend([0] * len(sentence[: sentence.index('[E1]')]))
            entity_mask.append(1)  # [E1]
            entity_mask.extend([0] * len(sentence[sentence.index('[E1]') + 1: sentence.index('[E2]')]))
            entity_mask.append(1)  # [E2]
            entity_mask.extend([0] * len(sentence[sentence.index('[E2]') + 1:]))
            entity_mask.append(0)  # => [SEP]
            entity_mask.extend([0] * (self.args.max_length - len(entity_mask)))

            sentence = ''.join(sentence)
            encodings_dict = self.tokenizer(sentence,
                                            truncation=True,
                                            max_length=self.args.max_length,
                                            padding='max_length')

            input_ids = encodings_dict['input_ids']
            attention_mask = encodings_dict['attention_mask']

            input_ids = torch.tensor(input_ids).unsqueeze(0)
            attention_mask = torch.tensor(attention_mask).unsqueeze(0)
            entity_mask = torch.BoolTensor(entity_mask).unsqueeze(0)

            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            entity_mask = entity_mask.to(DEVICE)

            vec_predict = self.model(input_idx=input_ids, attention_mask=attention_mask, entity_mask=entity_mask)[0]

            soft_predict = torch.softmax(vec_predict, dim=0)
            predict_prob, predict_index = torch.max(soft_predict.cpu().data, dim=0)

            i2label = {value: key for key, value in self.label2i.items()}

            predict_class = i2label[predict_index.item()]
            predict_prob = predict_prob.item()
            return predict_prob, predict_class

    def load(self):
        self.tokenizer = load_tokenizer(self.args.pretrained_model_path, self.SPECIAL_TOKEN)
        albertConfig = load_config(self.args.pretrained_model_path, self.tokenizer)
        albert_model = build_model(albertConfig)
        tag_num = len(self.label2i)
        self.model = AlbertFC(albertConfig, albert_model, tag_num)
        # self.model = torch.load(self.args.model_path, map_location=DEVICE)
        self.model.load_state_dict(torch.load(self.args.model_path, map_location=DEVICE))
        logger.info('loading model {}'.format(self.args.model_path))
        self.model = self.model.to(DEVICE)

    def test(self, test_path):
        test_dataset = GetDataset(test_path, self.tokenizer, self.args.max_length, self.SPECIAL_TOKEN, self.label2i)
        test_score = self._validate(test_dataset)
        return test_score

    def _validate(self, test_dataset):
        self.model.eval()
        with torch.no_grad():
            labels = np.array([])
            predicts = np.array([])
            for dev_item in tqdm(test_dataset):
                label = dev_item['label']
                input_ids = dev_item['input_ids']
                attention_mask = dev_item['attention_mask']
                entity_mask = dev_item['entity_mask']

                label = label.to(DEVICE)
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                entity_mask = entity_mask.to(DEVICE)

                out = self.model(input_idx=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0), entity_mask=entity_mask.unsqueeze(0))

                # p,r,f1 metrics
                prediction = torch.max(torch.softmax(out, dim=1), dim=1)[1]
                pred_y = prediction.cpu().data.numpy().squeeze()
                target_y = label.cpu().data.numpy()
                labels = np.append(labels, target_y)
                predicts = np.append(predicts, pred_y)

            report = get_score(labels, predicts)
            print('dev dataset len:{}'.format(len(test_dataset)))
            logger.info('dev_score: {}'.format(report))
        return report

    def validate(self, val_iter, criterion):
        self.model.eval()
        with torch.no_grad():
            labels = np.array([])
            predicts = np.array([])
            val_loss = 0.0
            for dev_item in tqdm(val_iter):
                label = dev_item['label']
                input_ids = dev_item['input_ids']
                attention_mask = dev_item['attention_mask']
                entity_mask = dev_item['entity_mask']

                label = label.to(DEVICE)
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                entity_mask = entity_mask.to(DEVICE)
                out = self.model(input_idx=input_ids, attention_mask=attention_mask, entity_mask=entity_mask)
                loss = criterion(out, label)

                val_loss += loss.item()

                # p,r,f1 metrics
                prediction = torch.max(torch.softmax(out, dim=1), dim=1)[1]
                pred_y = prediction.cpu().data.numpy().squeeze()
                target_y = label.cpu().data.numpy()
                labels = np.append(labels, target_y)
                predicts = np.append(predicts, pred_y)
            report = get_score(labels, predicts)

            print('dev dataset len:{}'.format(len(val_iter)))
            logger.info('dev_score: {}'.format(report))
        return val_loss / len(val_iter)
