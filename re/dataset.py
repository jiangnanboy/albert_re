from torch.utils.data import Dataset, DataLoader
import torch
from torchtext.data import BucketIterator
from sklearn.metrics import classification_report

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GetDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length, SPECIAL_TOKENS, label2i):
        '''
        build dataset
        :param data_path:
        :param tokenizer:
        :param max_length:
        :param SPECIAL_TOKENS:
        :param label2i:
        '''
        self.data_list = self.read_data(data_path)
        self.data_size = len(self.data_list)
        self.tokenizer = tokenizer
        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.max_length = max_length
        self.label2i = label2i

    def read_data(self, data_path):
        data_list = []
        with open(data_path, 'r', encoding='utf-8') as data_read:
            for line in data_read:
                line = line.strip().split()
                entity_1, entity_2, relation, sentence = tuple(line)
                if (entity_1 in sentence) and (entity_2 in sentence):
                    if (entity_1 not in entity_2) and (entity_2 not in entity_1):
                        data_list.append([entity_1, entity_2, relation, sentence])
        return data_list

    def __getitem__(self, idx):
        entity_1, entity_2, relation, sentence = self.data_list[idx]
        if sentence.index(entity_1) > sentence.index(entity_2): # order
            entity_1, entity_2 = entity_2, entity_1
        # construct： [E1] entity_1 [/E1], [E2] entity_2 [/E2]
        sentence = sentence.replace(entity_1, '[E1]' + entity_1 + '[/E1]', 1)
        sentence = sentence.replace(entity_2, '[E2]' + entity_2 + '[/E2]', 1)

        # entity mask: entity_1_start => 1 and entity_2_start => 1 => [0,0,1,0,0,0,1,0]
        sentence = ' '.join(list(sentence))
        sentence = sentence.replace('[ E 1 ]', '[E1]')
        sentence = sentence.replace('[ / E 1 ]', '[/E1]')
        sentence = sentence.replace('[ E 2 ]', '[E2]')
        sentence = sentence.replace('[ / E 2 ]', '[/E2]')
        sentence = sentence.split()
        entity_mask = [0] # => [CLS]
        entity_mask.extend([0] * len(sentence[ : sentence.index('[E1]')]))
        entity_mask.append(1) # [E1]
        entity_mask.extend([0] * len(sentence[sentence.index('[E1]') + 1 : sentence.index('[E2]')]))
        entity_mask.append(1) # [E2]
        entity_mask.extend([0] * len(sentence[sentence.index('[E2]') + 1 : ]))
        entity_mask.append(0) # => [SEP]
        entity_mask.extend([0] * (self.max_length - len(entity_mask)))

        label = self.label2i[relation]
        sentence = ''.join(sentence)
        encodings_dict = self.tokenizer(sentence,
                                        truncation=True,
                                        max_length=self.max_length,
                                        padding='max_length')

        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        return {'label': torch.tensor(label),
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask),
                'entity_mask': torch.BoolTensor(entity_mask)}

    def __len__(self):
        return self.data_size

def get_train_val_dataloader(batch_size, trainset, train_ratio):
    '''
    split trainset to train and val
    :param batch_size:
    :param trainset:
    :param train_ratio
    :return:
    '''

    train_size = int(train_ratio * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])

    trainloader = DataLoader(train_dataset,
                             batch_size=batch_size,
                             shuffle=True)

    valloader = DataLoader(val_dataset,
                           batch_size=batch_size,
                           shuffle=False)

    return trainloader, valloader, train_dataset, val_dataset

def get_dataloader(dataset, batch_size, shuffle=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def get_iterator(dataset: Dataset, batch_size, sort_key=lambda x: len(x.input_ids), sort_within_batch=True, shuffle=True):
    return BucketIterator(dataset, batch_size=batch_size, sort_key=sort_key,
                          sort_within_batch=sort_within_batch, shuffle=shuffle)

def get_score(labels, predicts):
    return classification_report(labels, predicts, target_names=None)