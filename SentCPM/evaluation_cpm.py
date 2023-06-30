import sys
import io, os
import numpy as np
import logging
import argparse
from prettytable import PrettyTable
import torch
from typing import Any, Dict, List, Tuple
from cpm_live.generation.bee import CPMBeeBeamSearch
from cpm_live.models import CPMBeeTorch, CPMBeeConfig,CPMBee
from cpm_live.tokenizers import CPMBeeTokenizer
from cpm_live.training_tasks.bee.pretrain import convert_data_to_id
from cpm_live.utils import pad
import torch.nn.functional as F
import bmtrain as bmt
# Set up logger
from opendelta import LoraModel
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, 
            help="Transformers' config name or path")
    parser.add_argument("--model_name", type=str, 
            help="Transformers' model name or path")
    parser.add_argument("--delta_name", type=str, 
            help="Transformers'delta name or path")
    parser.add_argument("--pooler", type=str, 
            choices=['cls', 'avg'], 
            default='cls', 
            help="Which pooler to use")
    parser.add_argument("--mode", type=str, 
            choices=['dev', 'test', 'fasttest'],
            default='test', 
            help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument("--task_set", type=str, 
            choices=['sts', 'transfer', 'full', 'na'],
            default='sts',
            help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
    parser.add_argument("--tasks", type=str, nargs='+', 
            default=['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                     'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
                     'SICKRelatedness', 'STSBenchmark'], 
            help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden")
    
    args = parser.parse_args()
    bmt.init_distributed(seed=1024)
    # Load transformers' model checkpoint
    config = CPMBeeConfig.from_json_file(args.config_name)
    
    tokenizer = CPMBeeTokenizer()
    
    model = CPMBee(config=config)
    bmt.load(model,args.model_name)
    delta_model = LoraModel(
            backbone_model=model, modified_modules=["project_q", "project_v"], backend="bmt"
    )
    delta_model.freeze_module(exclude=["deltas"], set_state_dict=True)
    # load checkpoints
    delta_state = torch.load(args.delta_name)
    model.load_state_dict(delta_state,strict=False)
    model.cuda()
    
    
    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness','CSTS']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness','CSTS']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                         'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                         'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return
    
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


    def _convert_to_tensors(data: Any, in_context_samples: List[Any] = []):
        answer_placeholders = []
        def _put_placeholder(data: Any, path: List[str] = []):
            if isinstance(data, dict):
                ret = {}
                for k, v in data.items():
                    ret[k] = _put_placeholder(v, path + [k])
                return ret
            else:
                answer_placeholders.append(path)
                return "<ans_{}>".format(len(answer_placeholders))
        
        data["<ans>"] = _put_placeholder(data["<ans>"])
        (
            input_ids,
            input_id_subs,
            context,
            segment_ids,
            segment_rel,
            n_segments,
            table_states,
        ) = convert_data_to_id(tokenizer, data, shuffle_answer=False, max_depth=8)
        
        sub_ans_map: Dict[int, int] = {}
        for fake_id, token_sub in table_states["token_id_table"]["<ans>"].items():
            token = table_states["ext_table"][fake_id]
            if token.startswith("<ans_") and token.endswith(">"):
                ans_id = int(token[5:-1])
                sub_ans_map[token_sub] = ans_id

        tmp_input_ids = []
        tmp_input_sub = []
        tmp_input_seg = []

        predict_segments: List[Tuple[int, int]] = []
        for i in range(input_ids.shape[0]):
            if context[i] == 0:
                if input_ids[i] == tokenizer.encoder["<ans>"]:
                    # is ans
                    # (segment_id, ans_id)
                    predict_segments.append((segment_ids[i], sub_ans_map[input_id_subs[i]]))
            else:
                tmp_input_ids.append(input_ids[i])
                tmp_input_sub.append(input_id_subs[i])
                tmp_input_seg.append(segment_ids[i])

        if len(predict_segments) == 0:
            raise ValueError("No answer to predict")

        input_ids = np.array(tmp_input_ids, dtype=np.int32)
        input_id_subs = np.array(tmp_input_sub, dtype=np.int32)
        context = np.full_like(tmp_input_ids, 1, dtype=np.int8)
        segment_ids = np.array(tmp_input_seg, dtype=np.int32)
        sample_ids = np.zeros(input_ids.shape, dtype=np.int32)
        segment_rel_offset = np.zeros(input_ids.shape, dtype=np.int32)
        num_segments = np.full(input_ids.shape, n_segments, dtype=np.int32)

        for i, sample in enumerate(in_context_samples):
            (
                sample_input_ids,
                sample_id_subs,
                _,
                sample_segments,
                sample_rel,
                n_segments,
                table_states,
            ) = convert_data_to_id(self.tokenizer, sample, table_states, max_depth=8)
            input_ids = np.concatenate([input_ids, sample_input_ids], axis=0)
            input_id_subs = np.concatenate([input_id_subs, sample_id_subs], axis=0)
            context = np.concatenate(
                [context, np.ones(sample_input_ids.shape, dtype=np.int8)], axis=0
            )
            segment_ids = np.concatenate([segment_ids, sample_segments], axis=0)
            segment_rel_offset = np.concatenate(
                [
                    segment_rel_offset,
                    np.full(sample_input_ids.shape, segment_rel.shape[0], dtype=np.int32),
                ],
                axis=0,
            )
            segment_rel = np.concatenate([segment_rel, sample_rel], axis=0)
            sample_ids = np.concatenate(
                [sample_ids, np.full(sample_input_ids.shape, i + 1, dtype=np.int32)], axis=0
            )
            num_segments = np.concatenate(
                [num_segments, np.full(sample_input_ids.shape, n_segments, dtype=np.int32)], axis=0
            )
        input_pos = np.arange(input_ids.shape[0], dtype=np.int32)

        return (
            input_ids,
            input_id_subs,
            input_pos,
            context,
            segment_ids,
            segment_rel_offset,
            segment_rel,
            sample_ids,
            num_segments,
            predict_segments,
            answer_placeholders,
            table_states["ext_table"],
            table_states["token_id_table"],
        )
    
    def _process_list(data_list: List[Any]):
        pack_tensor = []
        other_info = []
        segment_rel_pack = []

        batch_ext_table_map: Dict[Tuple[int, int], int] = {}
        batch_ext_table_ids: List[int] = []
        batch_ext_table_sub: List[int] = []

        for data in data_list:
            # print(data)
            (
                input_ids,
                input_id_subs,
                input_pos,
                context,
                segment_ids,
                segment_rel_offset,
                segment_rel,
                sample_ids,
                num_segments,
                predict_segments,
                answer_placeholders,
                ext_table,
                token_id_table,
            ) = _convert_to_tensors(data, [])
            rev_ext_table: Dict[int, str] = {}
            for token, mp in token_id_table.items():
                if token == "<ans>":
                    continue
                token_id = tokenizer.encoder[token]
                for fake_id, token_sub in mp.items():
                    if token_sub > 0:
                        if (token_id, token_sub) not in batch_ext_table_map:
                            batch_ext_table_map[(token_id, token_sub)] = (
                                len(batch_ext_table_ids) + tokenizer.vocab_size
                            )
                            batch_ext_table_ids.append(token_id)
                            batch_ext_table_sub.append(token_sub)
                        rev_ext_table[batch_ext_table_map[(token_id, token_sub)]] = ext_table[
                            fake_id
                        ]
                    else:
                        rev_ext_table[token_id] = ext_table[fake_id]
            pack_tensor.append(
                {
                    "input": torch.from_numpy(input_ids).unsqueeze(0),
                    "input_sub": torch.from_numpy(input_id_subs).unsqueeze(0),
                    "input_pos": torch.from_numpy(input_pos).unsqueeze(0),
                    "context": torch.from_numpy(context).unsqueeze(0),
                    "sample_idx": torch.from_numpy(sample_ids).unsqueeze(0),
                    "num_segments": torch.from_numpy(num_segments).unsqueeze(0),
                    "segment": torch.from_numpy(segment_ids).unsqueeze(0),
                    "segment_rel_offset": torch.from_numpy(segment_rel_offset).unsqueeze(0),
                }
            )
            segment_rel_pack.append(torch.from_numpy(segment_rel))
            other_info.append(
                {
                    "predict_segments": predict_segments,
                    "answer_placeholders": answer_placeholders,
                    "ext_table": rev_ext_table,
                }
            )

        keys = set(pack_tensor[0].keys())
        padded = {}
        for key in keys:
            padded[key] = pad(pack_tensor, key)

        max_num_rels = 0
        for rel in segment_rel_pack:
            max_num_rels = max(max_num_rels, rel.size(0))
        padded_rels = torch.zeros(len(segment_rel_pack), max_num_rels, dtype=torch.int32)
        for i, rel in enumerate(segment_rel_pack):
            padded_rels[i, : rel.size(0)] = rel
        padded["segment_rel"] = padded_rels
        padded["batch_ext_table_ids"] = torch.tensor(
            batch_ext_table_ids, dtype=torch.int32
        )
        padded["batch_ext_table_sub"] = torch.tensor(
            batch_ext_table_sub, dtype=torch.int32
        )

        # move to model device
        for k, v in padded.items():
            if isinstance(v, torch.Tensor):
                padded[k] = v.cuda()

        return padded, other_info

    

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]
        sentences = [{"input":sentence.replace('<','[').replace('>',']'),"<ans>":""} for sentence in sentences]
        
        
        #下面要进行数据处理，将batch转化为CPM-Bee的输入格式
        model_inputs, other_info = _process_list(data_list=sentences)
        
        batch_size = model_inputs["input"].size(0)
        input: torch.Tensor = (
            model_inputs["input"]
        )
        input_sub: torch.Tensor = (
            model_inputs["input_sub"]
        )
        input_pos: torch.Tensor = (
            model_inputs["input_pos"]
        )
        context: torch.Tensor = (
            model_inputs["context"]
        )
        sample_ids: torch.Tensor = (
            model_inputs["sample_idx"]
        )
        num_segments: torch.Tensor = (
            model_inputs["num_segments"]
        )
        segment: torch.Tensor = (
            model_inputs["segment"]
        )
        segment_rel_offset: torch.Tensor = (
            model_inputs["segment_rel_offset"]
        )
        segment_rel: torch.Tensor = (
            model_inputs["segment_rel"]
        )
        span = torch.zeros(input.shape)
        length = torch.tensor([len(data) for data in input])

        ext_table_ids: torch.Tensor = model_inputs["batch_ext_table_ids"]
        ext_table_sub: torch.Tensor = model_inputs["batch_ext_table_sub"]
        ext_table_ids_cpu = ext_table_ids.cpu()
        ext_table_sub_cpu = ext_table_sub.cpu()
        with torch.no_grad():
            logits, hidden_state = model(
                input=input.cuda(),
                input_sub=input_sub.cuda(),
                length=length.cuda(),
                context=context.cuda(),
                span = span.cuda(),
                sample_ids=sample_ids.cuda(),
                num_segments=num_segments.cuda(),
                segment=segment.cuda(),
                segment_rel_offset=segment_rel_offset.cuda(),
                segment_rel=segment_rel.cuda(),
                ext_table_ids=ext_table_ids.cuda(),
                ext_table_sub=ext_table_sub.cuda(),
            )
            
            # print(logits.shape)
            # print(hidden_state.shape)

        # Apply different poolers
        if args.pooler == 'cls':
            # There is a linear+activation layer after CLS representation
            pooler_result = hidden_state[:,0,:]
            pooler_result = F.normalize(pooler_result,p=2,dim=1)
            pooler_result = whitening(pooler_result)
            return pooler_result.cpu()
        elif args.pooler == "avg":
            pooler_result = torch.mean(hidden_state,dim=1)
            pooler_result = F.normalize(pooler_result,p=2,dim=1)
            pooler_result = whitening(pooler_result)
            return pooler_result.cpu()
        else:
            raise NotImplementedError

    results = {}

    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result
    
    # Print evaluation results
    if args.mode == 'dev':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STSBenchmark', 'SICKRelatedness','CSTS']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                scores.append("0.00")
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['devacc']))    
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

    elif args.mode == 'test' or args.mode == 'fasttest':
        print("------ %s ------" % (args.mode))

        task_names = []
        scores = []
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness','CSTS']:
            task_names.append(task)
            if task in results:
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)

        task_names = []
        scores = []
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]['acc']))    
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)


if __name__ == "__main__":
    main()
