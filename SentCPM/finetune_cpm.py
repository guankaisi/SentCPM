import time
from typing import Dict, List, Union
import torch
import bmtrain as bmt
import os
from opendelta import LoraModel
from cpm_live.arguments import get_args
from datasets import load_dataset
from cpm_live.models import CPMBee, CPMBeeConfig
from cpm_live.tokenizers import CPMBeeTokenizer
from cpm_live.utils import allgather_objects
from cpm_live.training_tasks.bee import FinetuneDataset
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer
import torch.nn.functional as F
from cpmcl_model import Similarity
from tqdm import tqdm


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained('/run/user/guankaisi/openbmb/cpm_1b',trust_remote_code=True)
    return tokenizer


def get_model(args):
    config = CPMBeeConfig.from_json_file(args.model_config)
    model = CPMBee(config)
    model.config = config
    if args.load is not None:
        bmt.load(model, args.load)
    else:
        bmt.init_parameters(model)
    
    # insert LoRA
    if args.use_delta:
        delta_model = LoraModel(
            backbone_model=model, modified_modules=["project_q", "project_v"], backend="bmt"
        )
        delta_model.freeze_module(exclude=["deltas"], set_state_dict=True)
        delta_state = torch.load('cpm_finetune/cpm-bee-1b-delta4.pt')
        delta_model.log()
        model.load_state_dict(delta_state,strict=False)
    return model


def get_optimizer(args, model):
    optimizer = bmt.optim.AdamOffloadOptimizer(
        model.parameters(), weight_decay=args.weight_decay
    )
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    lr_scheduler = bmt.lr_scheduler.Noam(
        optimizer,
        start_lr=args.lr,
        warmup_iter=args.warmup_iters,
        end_iter=args.lr_decay_iters,
        num_iter=args.start_step,
    )
    return lr_scheduler


def setup_model_and_optimizer(args):
    model = get_model(args)
    tokenizer = get_tokenizer(args)
    bmt.synchronize()
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmt.synchronize()
    optim_manager = bmt.optim.OptimManager(
        loss_scale=args.loss_scale,
        loss_scale_factor=2,
        loss_scale_steps=512,
    )
    optim_manager.add_optimizer(optimizer, lr_scheduler)
    return tokenizer, model, optimizer, lr_scheduler, optim_manager


def initialize():
    args = get_args(finetune=True)
    bmt.init_distributed(seed=args.seed)
    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
    return args


def see_memory(detail=False):
    if detail:
        res = torch.cuda.memory_summary()
    else:
        res = (
            round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024), 2),
            round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024), 2),
        )
    torch.cuda.reset_peak_memory_stats()
    return res


def add_mem_time(info, mem_usage, tim_usage):
    torch.cuda.synchronize()
    mem_usage[info] = see_memory()
    tim_usage[info] = time.time()
    return mem_usage, tim_usage


def finetune(
    args,
    tokenizer: AutoTokenizer,
    model: CPMBee,
    optimizer: bmt.optim.AdamOffloadOptimizer,
    lr_scheduler: bmt.lr_scheduler.WarmupLRScheduler,
    optim_manager: bmt.optim.OptimManager,
):

    average_time = bmt.utils.AverageRecorder()
    if model.config.dtype == torch.half:
        loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
    else:
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)

    if args.tensorboard is not None and bmt.rank() == 0:
        from torch.utils.tensorboard import SummaryWriter
        import distutils.version  # noqa: F401

        if not os.path.exists(args.tensorboard):
            os.makedirs(args.tensorboard)
        writer = SummaryWriter(log_dir=args.tensorboard)

    best_eval_loss, eval_loss_increase = 1e9, 0
    global_token_pass = 0.0
    global_steps = 0
    global_world_size = bmt.world_size()
    sim = Similarity(temp=0.05)
    # 加载数据集
    extension = args.dataset.split(".")[-1]
    if extension == 'txt':
        extension = 'text'
    if extension == 'csv':
        datasets = load_dataset(extension, data_files=args.dataset, cache_dir="./data/", delimiter="\t" if "tsv" in args.dataset else ",")
    else:
        datasets = load_dataset(extension, data_files=args.dataset, cache_dir="./data/")
    train_dataset = datasets['train']
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
    )

    # 开始训练
    train_state = torch.load('cpm_finetune/cpm-bee-1b-train_state.pt')
    epoch0 = train_state['epoch']
    iteration_0 = train_state['iteration']

    # optimizer.load_state_dict(torch.load('cpm_finetune/cpm-bee-1b-opt_state.pt'))
    # lr_scheduler.load_state_dict(torch.load('cpm_finetune/cpm-bee-1b-lr_state.pt'))
    for epoch in range(epoch0,args.epoch):
        last_data = None
        for iteration, data in enumerate(dataloader):
            if iteration < iteration_0:
                data = next(iter(dataloader))
                iteration += 1
                continue
            iteration = iteration + 1
            global_steps = global_steps + 1
            skip_this_batch = False
            # 处理数据相关的内容
            if 'text' in data.keys():
                data = {'sent0':data['text'],'sent1':data['text']}
            total = len(data['sent0'])
            for idx in range(total):
                if data['sent0'][idx] is None:
                    data['sent0'][idx] = " "
                if data['sent1'][idx] is None:
                    data['sent1'][idx] = " "
            sentences = data['sent0'] + data['sent1']
            if 'hard_neg' in data.keys():
                for idx in range(total):
                    if data['hard_neg'][idx] is None:
                        data['hard_neg'][idx] = " "
                sentences += data['hard_neg']
            sentences = [{'input':sentence.replace('<','[').replace('>',']'),'<ans>':''} for sentence in sentences]
            sent_features = tokenizer(
                    sentences,
                    return_tensors='pt',
                    padding=True,
            )
            sent_features['input'] = sent_features['input_ids']
            sent_features['input_sub'] = sent_features['input_id_subs']
            sent_features['ext_table_ids'] = sent_features['batch_ext_table_ids']
            sent_features['ext_table_sub'] = sent_features['batch_ext_table_sub']
            sent_features['segment'] = sent_features['segment_ids']
            del sent_features['input_pos']
            del sent_features['segment_ids']
            del sent_features['input_ids']
            del sent_features['input_id_subs']
            del sent_features['batch_ext_table_ids']
            del sent_features['batch_ext_table_sub']
            del sent_features['other_info']
            inputs = sent_features
            inputs['length'] = torch.Tensor([len(sent) for sent in sent_features['input']])
            inputs['span'] = torch.Tensor([[0]*len(sent) for sent in sent_features['input']])
            for k in inputs:
                inputs[k] = inputs[k].cuda()

            # 数据处理完毕
            # ===========
            optim_manager.zero_grad()
            mem_usage = {}
            tim_usage = {}
            mem_usage, tim_usage = add_mem_time("init", mem_usage, tim_usage)

            # ===========
            _, hidden_state = model(
                **inputs
            )
            
            # 这个hidden_state就是句子向量要进行处理
            pooler_output = torch.mean(hidden_state,dim=1)
            pooler_output = F.normalize(pooler_output,p=2,dim=1)
            batch_size = args.batch_size
            z1 = pooler_output[:batch_size,:]
            z2 = pooler_output[batch_size:batch_size*2,:]
            z3 = pooler_output[batch_size*2:,:]

            cos_sim = sim(z1.unsqueeze(1),z2.unsqueeze(0))
            z1_z3_cos = sim(z1.unsqueeze(1),z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim,z1_z3_cos],1)
            labels = torch.arange(cos_sim.size(0)).long().cuda()
            loss = loss_func(cos_sim,labels)
             # ===========
            optim_manager.backward(loss)
            mem_usage, tim_usage = add_mem_time("backward", mem_usage, tim_usage)

            # ===========
            grad_norm = optim_manager.clip_grad_norm(optimizer.param_groups, max_norm=1.0)
            optim_manager.step()
            mem_usage, tim_usage = add_mem_time("optim", mem_usage, tim_usage)

            # ==========
            iteration_time = tim_usage["optim"] - tim_usage["init"]
            average_time.record(iteration_time)


            if iteration % 2 == 0:
                bmt.print_rank(
                    (
                        "| Epoch: {:3d} | Iter: {:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | grad_norm: {:.4f} "
                    ).format(
                        epoch,
                        iteration,
                        loss.item(),
                        lr_scheduler.current_lr,
                        int(optim_manager.loss_scale),
                        grad_norm, 
                    )
                )

            if iteration % 100 == 1:
                state_dict = model.state_dict()
                train_state = {
                    'epoch':epoch,
                    'iteration':iteration
                }
                torch.save(train_state,os.path.join(args.save, args.save_name + "-train_state.pt"))
                torch.save(lr_scheduler.state_dict(),os.path.join(args.save, args.save_name + "-lr_state.pt"))
                torch.save(optimizer.state_dict(),os.path.join(args.save, args.save_name + "-opt_state.pt"))
                torch.save(state_dict, os.path.join(args.save, args.save_name + "-delta.pt"))

            


            
            


def main():
    args = initialize()
    tokenizer, model, optimizer, lr_scheduler, optim_manager = setup_model_and_optimizer(args)
    print("-"*20)
    finetune(args, tokenizer, model, optimizer, lr_scheduler, optim_manager)


if __name__ == "__main__":
    main()