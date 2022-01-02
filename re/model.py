import torch
import torch.nn as nn
from transformers import BertTokenizer, AlbertModel, AlbertConfig
import sys

sys.path.append('/home/shiyan/sy/albert_re/')

from ..utils.log import logger

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_tokenizer(model_path, special_token):
    logger.info('loading tokenizer {}'.format(model_path))
    tokenizer = BertTokenizer.from_pretrained(model_path)
    if special_token:
        tokenizer.add_special_tokens(special_token)
    return tokenizer

def load_config(pretrained_model_path, tokenizer):
    albertConfig = AlbertConfig.from_pretrained(pretrained_model_path,
                                                cls_token_id=tokenizer.cls_token_id,
                                                sep_token_id=tokenizer.sep_token_id,
                                                pad_token_id=tokenizer.pad_token_id,
                                                unk_token_id=tokenizer.unk_token_id,
                                                output_attentions=False,  # whether or not return [attentions weights]
                                                output_hidden_states=False)  # whether or not return [hidden states]
    return albertConfig

def load_pretrained_model(pretrained_model_path, tokenizer, special_token):
    logger.info('loading pretrained model {}'.format(pretrained_model_path))
    albertConfig = load_config(pretrained_model_path, tokenizer)
    model = AlbertModel.from_pretrained(pretrained_model_path, config=albertConfig)

    if special_token:
        # resize special token
        model.resize_token_embeddings(len(tokenizer))

    return model, albertConfig

def build_model(albertConfig):
    logger.info('build albertmodel!')
    model = AlbertModel(config=albertConfig)
    return model

class AlbertFC(nn.Module):
    def __init__(self, config, pretrained_model, tag_num):
        super(AlbertFC, self).__init__()
        self.config = config
        self.model = pretrained_model
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier1 = nn.Linear(2 * self.config.hidden_size, self.config.hidden_size)
        self.classifier2 = nn.Linear(self.config.hidden_size, tag_num)

    def forward(self, input_idx, token_type_ids=None, attention_mask=None, entity_mask=None):
        outputs = self.model(input_ids=input_idx, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0] # last hidden_size of albert => [batch_size, seq_len, hidden_size]
        # get entity head and tail hidden_size
        entity_head_tail_sequence_output = torch.masked_select(sequence_output, entity_mask.unsqueeze(-1))
        # reshape => [batch_size, 2*hidden_size]
        entity_head_tail_sequence_output = entity_head_tail_sequence_output.reshape(sequence_output.shape[0], 2 * sequence_output.shape[2])
        entity_head_tail_sequence_output = self.dropout(entity_head_tail_sequence_output)
        classifier1_out = self.classifier1(entity_head_tail_sequence_output)
        logits = self.classifier2(classifier1_out)
        return logits

