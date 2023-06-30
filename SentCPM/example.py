from transformers import AutoTokenizer
import bmtrain as bmt
from opendelta import LoraModel
from cpm_live.models import CPMBeeConfig,CPMBee
from cpm_live.tokenizers import CPMBeeTokenizer
from input_type import process_sentence_list
from cpm_live.utils import pad
import torch.nn.functional as F
import torch
from typing import Any, Dict, List, Tuple
import numpy as np

# 当batch_size >1时，进行白化
def whitening(input):
    # Step 1: 计算均值
    mean = torch.mean(input, dim=0)
    # Step 2: 计算标准差
    std = torch.std(input, dim=0)
    # Step 3: 中心化
    centered_tensor = input - mean
    # Step 4: 白化
    whitened_tensor = centered_tensor / std
    return whitened_tensor



if __name__ == "__main__":
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