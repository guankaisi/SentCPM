# Sentence Embedding for CPM-Bee

## 📖介绍

本仓库针对[CPM-Bee](https://github.com/OpenBMB/CPM-Bee)生成句子向量进行了相应的增量微调，并取得了很好的效果。

我们主要用SimCSE中有监督对比学习对[CPM-Bee](https://github.com/OpenBMB/CPM-Bee)进行增量微调，并在[SentEval](https://github.com/facebookresearch/SentEval) 上测试了测试了CPM-Bee生成的sentence Embedding效果, 同时为了测试中文效果，我们在SentEval的基础上添加了[Chinese STSBenchmark](https://github.com/pluto-junzeng/CNSD)测试数据，并取得了很好的效果

## 🚀 如何使用

1. 克隆仓库并进入源代码

```
git clone xxx/SentCPM
```

2.创建conda 环境

```shell
conda create -n sentCPM python=3.10 -y
conda activate sentCPM
```

3.安装依赖

```shell
pip install torch>=1.10
pip install -r requirements.txt
```

4.下载模型和delta权重到对应位置

5.参考`example.py`

加载模型和delta模型，将隐藏层充当其sentence-embedding

```python
bmt.init_distributed(seed=1024)
    # Load transformers' model checkpoint
    config = CPMBeeConfig.from_json_file('config/cpm-bee-1b.json')
    tokenizer = CPMBeeTokenizer()
    model = CPMBee(config=config)
    bmt.load(model,'cpm-bee-1b-ckpt.pt')
    delta_model = LoraModel(
            backbone_model=model, modified_modules=["project_q", "project_v"], backend="bmt"
    )
    delta_model.freeze_module(exclude=["deltas"], set_state_dict=True)
    delta_state = torch.load('cpm_finetune/cpm-bee-1b-delta.pt')
    model.load_state_dict(delta_state,strict=False)
    model.cuda()
    sentences = [
        'an impenetrable and insufferable ball of pseudo-philosophic twaddle .'
    ]
    # 构建成CPM-Bee的输入格式
    sentences = [sentence.replace('<','[').replace('>',']') for sentence in sentences]
    sentences_after = [{"input":sentence,"<ans>":""} for sentence in sentences]
    model_inputs, other_info = process_sentence_list(data_list=sentences_after)
    # 构建CPM-Bee输入格式
    with torch.no_grad():
        _, hidden_state = model(**model_inputs)
    sentence_embedding = torch.mean(hidden_state,dim=1)
    sentence_embedding = F.normalize(sentence_embedding,p=2,dim=1)
    if sentence_embedding.shape[0]>1:
        sentence_embedding = whitening(sentence_embedding)
    print("sentece: ",sentences)
    print("sentence_embedding: ",sentence_embedding)
```



## 👀对CPM进行增量微调

增量微调基于Opendelta，使用了Lora方法进行微调

参考[SimCSE](https://github.com/princeton-nlp/SimCSE)中有监督微调的方式

训练数据集为NLI数据，包括SNLI和MNLI数据集，可以使用以下脚本下载

```shell
cd data
bash download_nli.sh
```

具体微调代码，按照[CPM-Bee微调教程](https://github.com/OpenBMB/CPM-Bee/tree/main/tutorials/basic_task_finetune)改写

参见`finetune.py`

运行脚本为

```shell
#!/bin/bash
CUDA_VISIBLE_DEVICES= 0,7
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=1  --rdzv_endpoint=localhost:12345 finetune_cpm.py \
--model-config /run/user/guankaisi/config/cpm-bee-1b.json \
--load /run/user/guankaisi/cpm-bee-1b-ckpt.pt \
--dataset data/nli_for_simcse.csv \
--epoch 2 \
--batch-size 16 \
--lr 1e-5 \
--save cpm_finetune/ \
--save-name cpm-bee-1b \
--use-delta
```



## 🔗delta 模型

我们对cpm-1b和cpm-10b模型进行句向量微调，并开源出相应delta-model

| Model             | 基座模型 | 链接 |
| ----------------- | -------- | ---- |
| SentCPM-delta-1b  | CPM-1b   |      |
| SentCPM-delta-10b | CPM-10b  |      |

##  🌸如何测试

所有测试数据我们都基于[SentEval](https://github.com/facebookresearch/SentEval)，同时我们向基础的SentEval版本中添加了测试中文能力的数据集[CSTS-B](Chinese STSBenchmark), 可以使用同一套测试脚本进行测试

我们有两种类型的测试代码，一种是测试基于Huggingface的模型`evaluation.py`

测试脚本示例为

```shell
#！/bin/bash
CUDA_VISIBLE_DEVICES=3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=1 --rdzv_endpoint=localhost:12349 evaluation.py \
--model_name_or_path bert-base-uncased \
--pooler avg \
--task_set full \
--mode test \
```

另一种是专门测试CPM模型的测试程序`evaluation_cpm.py`

测试脚本示例为

```Shell
#!/bin/bash
python evaluation_cpm.py \
--model_name cpm-bee-1b-ckpt.pt \
--config_name config/cpm-bee-1b.json \
--delta_name cpm_finetune/cpm-bee-1b-delta.pt \
--pooler avg \
--task_set full \
--mode test \
```



## 💫 性能表现

我们主要测试了[SentEval](https://github.com/facebookresearch/SentEval)中的各个数据集，以及中文能力的Chinese STS-B

我们首先在语句相似度匹配STS系列数据集上进行测试，我们的模型取得了大于GPT-ada-embedding的效果

| Model                                               | STS12     | STS13     | STS14     | STS15     | STS16     | STSBenchmark | SICKRelatedness | Avg        |
| --------------------------------------------------- | --------- | --------- | --------- | --------- | --------- | ------------ | --------------- | ---------- |
| **simcse-CPM-1b**                                   | 75.24     | 84.81     | 79.85     | 82.29     | 79.99     | 83.85        | 77.42           | 80.49      |
| **simcse-CPM-10b**                                  | 76.84     | **87.37** | **82.21** | 82.78     | 82.66     | 86.48        | 79.79           | **82.59**  |
| GPT-embedding-002                                   | 71.08     | 81.85     | 76.37     | **86.03** | **85.60** | 84.30        | 80.25           | 80.78      |
| sup-simcse-robert                                   | **78.81** | 86.37     | 81.35     | 84.27     | 81.52     | **86.63  **  | **81.39  **     | **82.91 ** |
| sup-simcse-bert                                     | 75.17     | 82.73     | 77.52     | 85.74     | 80.84     | 82.95        | 80.56           | 80.79      |
| [m3e-base](https://huggingface.co/moka-ai/m3e-base) | 69.00     | 70.92     | 71.42     | 83.12     | 76.80     | 77.52        | 66.57           | 73.62      |
| bert-avg-whitening                                  | 30.87     | 59.89     | 47.73     | 60.29     | 63.73     | 47.29        | 58.22           | 52.57      |
| bert-cls-whitening                                  | 21.54     | 32.11     | 21.28     | 37.89     | 44.24     | 20.29        | 42.42           | 31.40      |

在Transfer数据集上效果

| Model                                              | MR        | CR        | SUBJ      | MPQA      | SST       | TREC      | MRPC  | Avg.      |
| -------------------------------------------------- | --------- | --------- | --------- | --------- | --------- | --------- | ----- | --------- |
| simcse-cpm-1b                                      | 85.10     | 90.28     | 94.43     | 90.53     | 90.01     | 91.20     | 76.46 | 88.29     |
| simcse-cpm-10b                                     | **89.22** | 92.21     | **95.12** | **91.34** | **93.47** | 91.20     | 75.13 | **89.67** |
| avg-bert                                           | 78.66     | 86.25     | 94.37     | 88.66     | 84.40     | **92.80** | 69.54 | 84.94     |
| SimCSE-RoBERT large                                | 88.12     | **92.37** | 95.11     | 90.49     | 92.75     | 91.80     | 76.64 | 89.61     |
| [m3ebase](https://huggingface.co/moka-ai/m3e-base) | 71.67     | 80.55     | 88.02     | 81.56     | 72.27     | 85.40     | 70.84 | 78.62     |
| GPT-embedding-002                                  |           |           |           |           |           |           |       |           |

Chinese STS-Benchmark

| 模型                      | Chinese-STS-B-dev | Chinese-STS-B-test |
| ------------------------- | ----------------- | ------------------ |
| bert_avg                  | 0.2549            | 0.2059             |
| sup_simcse_robert（SNLI） | 0.7499            | 0.6909             |
| simcse-cpm-1b             | 0.838             | 0.7743             |
| **simcse-cpm-10b**        | **0.836**         | **0.7936**         |
| m3e-base                  | 0.8245            | 0.7753             |



## 引用

```latex
@misc{sentCPM,
  author = {},
  title = {},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {}
}
```







