import pickle
import time
import numpy as np
import utils.utils as utils
from utils.utils import *
from utils.loss_utils import *
# from dataset_utils import *
from utils.dist_utils import *
from dataset.dataset_contrastive import *
from dataset.dataset_aff import *
import torch.nn as nn
import torch
import time
import os
# os.environ['CUDA_LAUNCH_BLOCKING']='1'
import argparse
import time
from torch.utils.data import DataLoader          
from prefetch_generator import BackgroundGenerator
from model.equiscore import EquiScore
from equiformer.nets.graph_attention_transformer import GraphAttentionTransformer_dgl
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())    
now = time.localtime()
from rdkit import RDLogger
from scipy.stats import spearmanr,pearsonr
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import json


def run(local_rank,args):
    
    args.local_rank = local_rank
    args.lmdb_cache = "lmdbs/affTest100_draw"
    with open("affTest100.json", 'r') as f:
        test_keys = json.load(f)
    test_keys = [key for key in test_keys if "Amikacin" in key]
    with open("best_keys.json", 'r') as f:
        test_keys = json.load(f)   
    with open("worst_keys.json", 'r') as f:
        test_keys = json.load(f)      
    test_keys  = test_keys["dimethyl"]
    # dimethyl   Sisomicinin Netilmicincin Mitoxantronetrone Amikacinn
    model = EquiScore(args) if args.model == 'EquiScore' else GraphAttentionTransformer_dgl()
    
    print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    # args.device = args.local_rank
    args.device ='cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(args.device)


    test_dataset = ESDataset_aff(test_keys,args, args.data_path,args.debug)

    test_sampler = SequentialDistributedSampler(test_dataset,args.batch_size) if args.ngpu > 1 else None
    #    use sampler to balance the training data or not 


    test_dataloader = DataLoaderX(test_dataset, args.batch_size, sampler=test_sampler,\
        shuffle=False, num_workers = args.num_workers, collate_fn=test_dataset.collate,pin_memory=True,prefetch_factor = 4) 


    i=3
    # 2024-11-24-13-32-55
    for i in range(10):
        best_name = "workdir/contrastive/2025-01-12-16-36-33" + f'/save_best_model_{i}'+'.pt'
        if not os.path.exists(best_name):
            best_name =  "workdir/contrastive/2025-01-12-16-36-33" + f'/save_early_stop_model_{i}'+'.pt'     
        checkpoint = torch.load(best_name)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        with torch.no_grad():
            test_losses,test_true,test_pred,keys = [], [],[],[]
            for i_batch, (g,full_g,Y,key) in tqdm.tqdm(enumerate(test_dataloader),total = len(test_dataloader)):
                # print(Y)
                # print(key)
                # print(full_g)
                # print(g)
                model.zero_grad()
                g = g.to(args.device,non_blocking=True)
                full_g = full_g.to(args.device,non_blocking=True)
            
                pred = model(g,full_g)
            


                keys.extend(key)
                if pred.shape[1]==2:
                    pred = torch.softmax(pred,dim = -1)[:,1]
                test_pred.append(pred.data) if args.ngpu > 1 else test_pred.append(pred.data)

            # gather ngpu result to single tensor
            if args.ngpu > 1:
                
                test_pred = distributed_concat(torch.concat(test_pred, dim=0), 
                                                len(test_sampler.dataset)).cpu().numpy()
            else:
                test_pred = torch.concat(test_pred, dim=0).cpu().numpy()
            test_pred = np.array(test_pred).ravel()
            keys = np.array(keys).ravel()
            # 将数组按顺序每20个一组进行重塑
            N = 20
            reshaped_test_pred = test_pred.reshape(-1, N)
            print(reshaped_test_pred,keys)
            # 对每组取平均值
            mean_test_pred = np.mean(reshaped_test_pred, axis=1)
            max_test_pred = np.min(reshaped_test_pred, axis=1)
            # 打印结果
            print(f"Mean of each group: {mean_test_pred}")
            print(f"Max of each group: {max_test_pred}")
            continue
        return
        # 划分为5组，每组20个样本（同时处理 key 和 test_pred）
        group_size = N
        groups = [(keys[i * group_size: (i + 1) * group_size], test_pred[i * group_size: (i + 1) * group_size]) for i in range(5)]

        # 结果存储
        selected_samples = []
        selected_keys = []
        nums = 20
        # 遍历每组，按规则选取样本
        for i, (group_keys, group_values) in enumerate(groups):
            if i == 1:  # 第二组，选取最小的5个值
                sorted_indices = np.argsort(group_values)[:nums]  # 最小值的索引
            elif i == 2:  # 第三组，选取最大的5个值
                sorted_indices = np.argsort(group_values)[-nums:]  # 最大值的索引
            else:  # 其余组，随机选取5个值
                sorted_indices = np.argsort(group_values)[:-10]  # 去掉最大最小的值
                print(sorted_indices)
                sorted_indices = np.random.choice(len(sorted_indices), size=nums, replace=False)  # 随机索引
            
            # 根据索引记录选取的 key 和 value
            selected_keys.append(group_keys[sorted_indices])
            selected_samples.append(group_values[sorted_indices])

        # 输出结果
        for i, (keys, values) in enumerate(zip(selected_keys, selected_samples)):
            print(f"Group {i + 1}:")
            print(f"  Selected Keys: {keys}")
            print(f"  Selected Values: {values}")
            print(f"  Mean: {np.mean(values):.3f}")


if '__main__' == __name__:
    '''distribution training'''
    from torch import distributed as dist
    import torch.multiprocessing as mp
    from utils.dist_utils import *
    from utils.parsing import parse_train_args
    # get args from parsering function
    args = parse_train_args()
    # set gpu to use
    # if args.ngpu>0:
    #     cmd = get_available_gpu(num_gpu=args.ngpu, min_memory=8000, sample=3, nitro_restriction=False, verbose=True)
    #     if cmd[-1] == ',':
    #         os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]
    #     else:
    #         os.environ['CUDA_VISIBLE_DEVICES']=cmd
    os.environ["MASTER_ADDR"] = args.MASTER_ADDR
    os.environ["MASTER_PORT"] = args.MASTER_PORT
    # run_test3(0,args)
    
    # exit()
    
    from torch.multiprocessing import Process
    world_size = args.ngpu

    # use multiprocess to train
    processes = []
    for rank in range(world_size):
        p = Process(target=run, args=(rank, args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


