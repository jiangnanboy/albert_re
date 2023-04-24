# albert-fc for RE(Relation Extraction)，中文关系抽取

## 概述
关系抽取是指从非结构化文本中抽取语义关系的一项基本任务。提取出来的关系通常发生在两个或多个特定类型的实体之间(例如，人、组织、地点等)，
比如在人际之间的关系有同门、朋友、夫妻、同事、父母、上下级等。

![image](https://raw.githubusercontent.com/jiangnanboy/albert_re/master/image/example.png)


## 方法

利用huggingface/transformers中的albert+fc进行中文句子关系分类。

利用albert加载中文预训练模型，后接一个前馈分类网络。利用albert预训练模型进行fine-tune。

albert预训练模型下载：

链接：https://pan.baidu.com/s/1GSuwd6z_YG1oArnHVuN0eQ 

提取码：sy12

整个流程是：

- 数据经albert后获取最后的隐层hidden_state=768
- 根据albert的last hidden_state=768，将头实体和尾实体的hidden_state提取，并拼接，最后维度是（2 * 768）经一层前馈网络进行分类

![image](https://raw.githubusercontent.com/jiangnanboy/albert_re/master/image/albert-re.png)

 ## 数据说明

数据形式见data/

训练数据示例如下，其中各列为`头实体`、`尾实体`、`关系`、`句子`。

```
李敖	王尚勤	夫妻	李敖后来也认为，“任何当过王尚勤女朋友的人，未来的婚姻都是不幸的！
傅家俊	丁俊晖	好友	改写23年历史2010年10月29日，傅家俊1-5输给丁俊晖，这是联盟杯历史上首次出现了中国德比，丁俊晖傅家俊携手改写了
梁左	梁天	兄弟姐妹	-简介梁左与丈夫英达梁欢，女，梁欢和梁天的妹妹，英达的现任妻子。
```

## 训练和预测见（examples/test_re.py）

```
    re = RE(args)
    if train_bool(args.train):
        re.train()
        '''
        epoch: 58, acc_loss: 0.0080563764
        dev_score: 0.67
        val_loss: 0.06119863061138485, best_val_loss: 0.055413116525741
        '''
    else:
        re.load()
        # ner.test(args.test_path)
        # 钱钟书	辛笛	同门	与辛笛京沪唱和聽钱钟书与钱钟书是清华校友，钱钟书高辛笛两班。
        print(re.predict(text=('钱钟书', '辛笛', '与辛笛京沪唱和聽钱钟书与钱钟书是清华校友，钱钟书高辛笛两班。')))
        # (0.9997676014900208, '同门')

        # 平儿	贾琏	夫妻	此外，如贾琏偷娶尤二姐事，平儿虽然告诉了凤姐，但她对凤姐虐待尤二姐、害死尤二姐一事并不赞成
        print(re.predict(text=('平儿', '贾琏', '此外，如贾琏偷娶尤二姐事，平儿虽然告诉了凤姐，但她对凤姐虐待尤二姐、害死尤二姐一事并不赞成')))
        # (0.9999169111251831, '夫妻')
```

## 项目结构
- data
    - example.dev
    - example.train
- examples
    - test_re.py #训练及预测
- model
    - pretrained_model #存放预训练模型和相关配置文件
        - config.json
        - pytorch_model.bin
        - vocab.txt
- re
    - dataset.py
    - model.py
    - module.py
- utils
    - log.py

## 参考
- [transformers](https://github.com/huggingface/transformers)
- [Matching the Blanks: Distributional Similarity for Relation Learning](https://arxiv.org/pdf/1906.03158.pdf)

## contact

如有搜索、推荐、nlp以及大数据挖掘等问题或合作，可联系我：

1、我的github项目介绍：https://github.com/jiangnanboy

2、我的博客园技术博客：https://www.cnblogs.com/little-horse/

3、我的QQ号:2229029156
