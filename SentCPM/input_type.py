from cpm_live.training_tasks.bee.pretrain import convert_data_to_id
from cpm_live.utils import pad
import torch.nn.functional as F
from cpm_live.tokenizers import CPMBeeTokenizer
import torch
from typing import Any, Dict, List, Tuple
import numpy as np


def _convert_to_tensors(data: Any, in_context_samples: List[Any] = []):
    tokenizer = CPMBeeTokenizer()
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

def process_sentence_list(data_list: List[Any]):
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
                "context": torch.from_numpy(context).unsqueeze(0),
                "sample_ids": torch.from_numpy(sample_ids).unsqueeze(0),
                "num_segments": torch.from_numpy(num_segments).unsqueeze(0),
                "segment": torch.from_numpy(segment_ids).unsqueeze(0),
                "segment_rel_offset": torch.from_numpy(segment_rel_offset).unsqueeze(0),
                "length" : torch.tensor([len(data) for data in torch.from_numpy(input_ids).unsqueeze(0)])
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
    padded["ext_table_ids"] = torch.tensor(
        batch_ext_table_ids, dtype=torch.int32
    )
    padded["ext_table_sub"] = torch.tensor(
        batch_ext_table_sub, dtype=torch.int32
    )
    padded['span'] = torch.zeros(padded['input'].shape).cuda()
    # move to model device
    for k, v in padded.items():
        if isinstance(v, torch.Tensor):
            padded[k] = v.cuda()

    return padded, other_info