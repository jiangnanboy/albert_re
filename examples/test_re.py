import os
import argparse
import sys
sys.path.append('/home/sy/project/albert_re/')

from re.module import RE
if __name__ == '__main__':
    path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    print("Base path : {}".format(path))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--pretrained_model_path',
        default=os.path.join(path, 'model/pretrained_model'),
        type=str,
        required=False,
        help='The path of pretrained model!'
    )
    parser.add_argument(
        "--model_path",
        default=os.path.join(path, 'model/re_pytorch_model.bin'),
        type=str,
        required=False,
        help="The path of model!",
    )
    parser.add_argument(
        '--SPECIAL_TOKEN',
        default={"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]", "mask_token": "[MASK]",
                 "additional_special_tokens":["[E1]", "[/E1]", "[E2]", "[/E2]"]},
        type=dict,
        required=False,
        help='The dictionary of special tokens!'
    )
    parser.add_argument(
        '--LABEL2I',
        default={'unknown': 0,
                '父母': 1,
                '夫妻': 2,
                '师生': 3,
                '兄弟姐妹': 4,
                '合作': 5,
                '情侣': 6,
                '祖孙': 7,
                '好友': 8,
                '亲戚': 9,
                '同门': 10,
                '上下级': 11},
        type=dict,
        required=False,
        help='The dictionary of label2i!'
    )
    parser.add_argument(
        "--train_path",
        default=os.path.join(path, 'data/example.train'),
        type=str,
        required=False,
        help="The path of training set!",
    )
    parser.add_argument(
        '--dev_path',
        default=os.path.join(path, 'data/example.dev'),
        type=str,
        required=False,
        help='The path of dev set!'
    )
    parser.add_argument(
        '--test_path',
        default=None,
        type=str,
        required=False,
        help='The path of test set!'
    )
    parser.add_argument(
        '--log_path',
        default=None,
        type=str,
        required=False,
        help='The path of Log!'
    )
    parser.add_argument("--epochs", default=100, type=int, required=False, help="Epochs!")
    parser.add_argument(
        "--batch_size", default=64, type=int, required=False, help="Batch size!"
    )
    parser.add_argument('--step_size', default=50, type=int, required=False, help='lr_scheduler step size!')
    parser.add_argument("--lr", default=0.0001, type=float, required=False, help="Learning rate!")
    parser.add_argument('--clip', default=5, type=float, required=False, help='Clip!')
    parser.add_argument("--weight_decay", default=0, type=float, required=False, help="Regularization coefficient!")
    parser.add_argument(
        "--max_length", default=200, type=int, required=False, help="Maximum text length!"
    )
    parser.add_argument('--train', default='flase', type=str, required=False, help='Train or predict!')
    args = parser.parse_args()
    train_bool = lambda x:x.lower() == 'true'
    re = RE(args)
    if train_bool(args.train):
        re.train()
    else:
        re.load()
        # ner.test(args.test_path)
        # 钱钟书	辛笛	同门	与辛笛京沪唱和聽钱钟书与钱钟书是清华校友，钱钟书高辛笛两班。
        print(re.predict(text=('钱钟书', '辛笛', '与辛笛京沪唱和聽钱钟书与钱钟书是清华校友，钱钟书高辛笛两班。')))
        # 平儿	贾琏	夫妻	此外，如贾琏偷娶尤二姐事，平儿虽然告诉了凤姐，但她对凤姐虐待尤二姐、害死尤二姐一事并不赞成
        print(re.predict(text=('平儿', '贾琏', '此外，如贾琏偷娶尤二姐事，平儿虽然告诉了凤姐，但她对凤姐虐待尤二姐、害死尤二姐一事并不赞成')))
