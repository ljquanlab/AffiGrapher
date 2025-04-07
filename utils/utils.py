import numpy as np
import torch
import os
import sys
import pandas as pd
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from torch import distributed as dist
import os.path
import time
import torch.nn as nn
from rdkit.ML.Scoring.Scoring import CalcBEDROC
from collections import defaultdict
from sklearn.metrics import roc_auc_score,confusion_matrix,roc_curve
from sklearn.metrics import accuracy_score,auc,balanced_accuracy_score
from sklearn.metrics import recall_score,precision_score,precision_recall_curve
from sklearn.metrics import confusion_matrix,f1_score
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
from model.equiscore import conLoss
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())                            
# from dataset.dataset import ESDataset,DTISampler
from utils.dist_utils import *
N_atom_features = 28
from scipy.spatial import distance_matrix
import torch.nn.functional as F
import dgl
import matplotlib.pyplot as plt
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
def train_contrastive_gan(model, args, optimizer, loss_fn, train_dataloader, scheduler):
    # collect losses of each iteration
    train_losses = [] 
    loss_logits = []
    loss_conts = []
    gan_losses = []
    epoch_start = 0
    discriminator = model.discriminator
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    d_scheduler = torch.optim.lr_scheduler.OneCycleLR(d_optimizer, max_lr=args.max_lr,pct_start=args.pct_start,\
             steps_per_epoch=len(train_dataloader), epochs=args.epoch,last_epoch = -1 if len(train_dataloader)*epoch_start == 0 else len(train_dataloader)*epoch_start )
    model.train()
    discriminator.train()
    # alpha=1.0
    for i_batch, (g_batch, full_g_batch, Y) in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        optimizer.zero_grad()
        # d_optimizer.zero_grad()
        model.zero_grad()
        discriminator.zero_grad()

        g_batch = g_batch.to(args.device, non_blocking=True)
        full_g_batch = full_g_batch.to(args.device, non_blocking=True)
        Y = Y.to(args.device, non_blocking=True)
        Y = Y.unsqueeze(-1)

        # 模型前向传播，得到所有样本的嵌入
        embeddings, logits = model(g_batch, full_g_batch, contrastive=True)

        # 计算logit损失
        loss_logit = loss_fn(logits, Y)
        # PCC损失
        pcc_loss_value = pcc_loss(logits, Y)
        # SPCC损失
        spcc_loss_value = spcc_loss(logits, Y)

        # 拆分输出
        batch_size = args.batch_size
        pos1, pos2 = embeddings[0:batch_size], embeddings[batch_size:2*batch_size]
        neg1, neg2 = embeddings[2*batch_size:3*batch_size], embeddings[3*batch_size:]

        # 计算对比学习损失
        pos_sim = F.cosine_similarity(pos1, pos2)
        neg_sim = F.cosine_similarity(pos1, neg1)
        loss_cont = contrastive_loss(pos_sim, neg_sim)

        # GAN判别器损失
        real_pairs = torch.cat([pos1, pos2], dim=1)
        fake_pairs = torch.cat([pos1, neg1], dim=1)
        real_labels = torch.ones(batch_size, 1).to(args.device)
        fake_labels = torch.zeros(batch_size, 1).to(args.device)

        real_output = discriminator(real_pairs.detach())
        fake_output = discriminator(fake_pairs.detach())

        d_loss_real = F.binary_cross_entropy(real_output, real_labels)
        d_loss_fake = F.binary_cross_entropy(fake_output, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        # 更新判别器
        # d_loss.backward(retain_graph=True)
        # d_optimizer.step()

        # 生成器损失（对比学习损失 + GAN损失）
        g_loss_fake = F.binary_cross_entropy(fake_output, real_labels)
        alpha = 0.0
        loss1 = alpha * loss_cont + (1 - alpha) * loss_logit
        loss = args.mse_weight * loss1 - args.pcc_weight * pcc_loss_value - args.spcc_weight * spcc_loss_value + args.gan_weight * d_loss

        # 反向传播并更新模型参数
        # 反向传播并更新模型参数
        loss.backward()
        optimizer.step()

        # 打印损失
        train_losses.append(loss.item())
        loss_conts.append(loss_cont.item())
        loss_logits.append(loss_logit.item())
        gan_losses.append(d_loss.item())
        if args.lr_decay:
            scheduler.step()
            # d_scheduler.step()
        torch.cuda.empty_cache()
    return model, train_losses, loss_conts, loss_logits, optimizer, scheduler, gan_losses
def get_args_from_json(json_file_path, args_dict):
    """"
    docstring:
        use this function to update the args_dict from a json file if you want to use a json file save parameters 
    input:
        json_file_path: string
            json file path
        args_dict: args dict
            dict

    output:
        args dict
    """

    import json
    summary_filename = json_file_path
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)
    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]
    return args_dict

def initialize_model(model, device, args,load_save_file = False,init_classifer = True):
    """ initialize the model parameters or load the model from a saved file"""
    for param in model.parameters():
        if param.dim() == 1:
            continue
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
            

    if load_save_file:
        state_dict = torch.load(load_save_file,map_location = 'cpu')
        model_dict = state_dict['model']
        model_state_dict = model.state_dict()
        model_dict = {k:v for k,v in model_dict.items() if k in model_state_dict}
        model_state_dict.update(model_dict)
        model.load_state_dict(model_state_dict) 
        
        optimizer =state_dict['optimizer']
        epoch = state_dict['epoch']
        print('load save model!')
    if device:
        model = model.to(device)
    elif torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
      
        model = model.cuda(args.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                     device_ids=[args.local_rank], 
                                                     output_device=args.local_rank, 
                                                     find_unused_parameters=True, 
                                                     broadcast_buffers=False)
        if load_save_file:
            return model ,optimizer,epoch
        return model
    # model.to(args.local_rank)
    if load_save_file:
        return model ,optimizer,epoch
    return model

def get_logauc(fp, tp, min_fp=0.001, adjusted=False):
    """"
    docstring:
        use this function to calculate logauc 
    input:
        fp: list
            false positive
        tp: list
            true positive

    output: float
        logauc
    """
    
    lam_index = np.searchsorted(fp, min_fp)
    y = np.asarray(tp[lam_index:], dtype=np.double)
    x = np.asarray(fp[lam_index:], dtype=np.double)
    if (lam_index != 0):
        y = np.insert(y, 0, tp[lam_index - 1])
        x = np.insert(x, 0, min_fp)

    dy = (y[1:] - y[:-1])
    with np.errstate(divide='ignore'):
        intercept = y[1:] - x[1:] * (dy / (x[1:] - x[:-1]))
        intercept[np.isinf(intercept)] = 0.
    norm = np.log10(1. / float(min_fp))
    areas = ((dy / np.log(10.)) + intercept * np.log10(x[1:] / x[:-1])) / norm
    logauc = np.sum(areas)
    if adjusted:
        logauc -= 0.144620062  # random curve logAUC
    return logauc

def get_metrics(train_true,train_pred):
    # lr_decay
    """"
    docstring:
        calculate the metrics for the dataset
    input:
        train_true: list
            label
        train_pred: list
            predicted label

    output: list
        metrics
    """
    try:
        train_pred = np.concatenate(np.array(train_pred,dtype=object), 0).astype(np.float)
        train_true = np.concatenate(np.array(train_true,dtype=object), 0).astype(np.float)
    except:
        pass
    train_pred_label = np.where(train_pred > 0.5,1,0).astype(np.float)

    tn, fp, fn, tp = confusion_matrix(train_true,train_pred_label).ravel()
    train_auroc = roc_auc_score(train_true, train_pred) 
    train_acc = accuracy_score(train_true,train_pred_label)
    train_precision = precision_score(train_true,train_pred_label)
    train_sensitity = tp/(tp + fn)
    train_specifity = tn/(fp+tn)
    ps,rs,_ = precision_recall_curve(train_true,train_pred)
    train_auprc = auc(rs,ps)
    train_f1 = f1_score(train_true,train_pred_label)
    train_balanced_acc = balanced_accuracy_score(train_true,train_pred_label)
    fp,tp,_ = roc_curve(train_true,train_pred)
    train_adjusted_logauroc = get_logauc(fp,tp)
    # BEDROC
    sort_ind = np.argsort(train_pred)[::-1] # Descending order
    BEDROC = CalcBEDROC(train_true[sort_ind].reshape(-1, 1), 0, alpha = 80.5)
    return train_auroc,BEDROC,train_adjusted_logauroc,train_auprc,train_balanced_acc,train_acc,train_precision,train_sensitity,train_specifity,train_f1


def random_split(train_keys, split_ratio=0.9, seed=0, shuffle=True):
    """
    docstring:
        split the dataset into train and validation set by random sampling, this function not useful for new target protein prediction
    """
    
    
    dataset_size = len(train_keys)
    """random splitter"""
    np.random.seed(seed)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(split_ratio * dataset_size)
    train_idx, valid_idx = indices[:split], indices[split:]
    return [train_keys[i] for i in train_idx], [train_keys[i] for i in valid_idx]

def evaluator(model,loader,loss_fn,args,test_sampler):
    model.eval()
    with torch.no_grad():
        test_losses,test_true,test_pred,keys = [], [],[],[]
        for i_batch, (g,full_g,Y,key) in enumerate(loader):
            
            model.zero_grad()
            g = g.to(args.device,non_blocking=True)
            full_g = full_g.to(args.device,non_blocking=True)
            Y = Y.to(args.device,non_blocking=True)
            Y = Y.unsqueeze(-1)
            pred,score_matrix = model(g,full_g,getScore=True)
            loss = loss_fn(pred ,Y) 

            if args.ngpu > 1:
                dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
                loss /= float(dist.get_world_size()) # get all loss value 
            # collect loss, true label and predicted label
            test_losses.append(loss.data)
            if args.ngpu > 1:
                test_true.append(Y.data)
            else:
                test_true.append(Y.data)
            keys.extend(key)
            if pred.shape[1]==2:
                pred = torch.softmax(pred,dim = -1)[:,1]
            pred = pred if args.loss_fn == 'auc_loss' else pred
            test_pred.append(pred.data) if args.ngpu > 1 else test_pred.append(pred.data)

        # gather ngpu result to single tensor
        if args.ngpu > 1:
            test_true = distributed_concat(torch.concat(test_true, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
            test_pred = distributed_concat(torch.concat(test_pred, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
        
        else:
            test_true = torch.concat(test_true, dim=0).cpu().numpy()
            test_pred = torch.concat(test_pred, dim=0).cpu().numpy()
    return test_losses,test_true,test_pred,keys
def evaluator_aff(model,loader,loss_fn,args,test_sampler):
    model.eval()
    with torch.no_grad():
        test_losses,test_true,test_pred,keys = [], [],[],[]
        for i_batch, (g,full_g,Y,key) in enumerate(loader):
            
            model.zero_grad()
            g = g.to(args.device,non_blocking=True)
            full_g = full_g.to(args.device,non_blocking=True)
            Y = Y.to(args.device,non_blocking=True)
            Y = Y.unsqueeze(-1)
            pred = model(g,full_g)
            
            mse_loss = loss_fn(pred, Y)
            # print(mse_loss)
            # PCC损失
            pcc_loss_value = pcc_loss(pred, Y)

            # SPCC损失
            spcc_loss_value = spcc_loss(pred, Y)

            # 组合损失 (可以根据需要调整权重)
            loss = args.mse_weight * mse_loss 

            if args.ngpu > 1:
                dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
                loss /= float(dist.get_world_size()) # get all loss value 
            # collect loss, true label and predicted label
            test_losses.append(loss.data)
            if args.ngpu > 1:
                test_true.append(Y.data)
            else:
                test_true.append(Y.data)
            keys.extend(key)
            if pred.shape[1]==2:
                pred = torch.softmax(pred,dim = -1)[:,1]
            pred = pred if args.loss_fn == 'auc_loss' else pred
            test_pred.append(pred.data) if args.ngpu > 1 else test_pred.append(pred.data)

        # gather ngpu result to single tensor
        if args.ngpu > 1:
            test_true = distributed_concat(torch.concat(test_true, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
            test_pred = distributed_concat(torch.concat(test_pred, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
        
        else:
            test_true = torch.concat(test_true, dim=0).cpu().numpy()
            test_pred = torch.concat(test_pred, dim=0).cpu().numpy()
    return test_losses,test_true,test_pred,keys
def evaluator_aff_equi(model,loader,loss_fn,args,test_sampler):
    model.eval()
    with torch.no_grad():
        test_losses,test_true,test_pred,keys = [], [],[],[]
        for i_batch, (g,full_g,Y,key) in enumerate(loader):
            batch = g.batch_num_nodes()
            model.zero_grad()
            g = g.to(args.device,non_blocking=True)
            full_g = full_g.to(args.device,non_blocking=True)
            Y = Y.to(args.device,non_blocking=True)
            Y = Y.unsqueeze(-1)
            batch = torch.arange(len(Y)).repeat_interleave(batch).to(args.device)
            
            pred = model(g,batch)
            loss = loss_fn(pred ,Y) 

            if args.ngpu > 1:
                dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
                loss /= float(dist.get_world_size()) # get all loss value 
            # collect loss, true label and predicted label
            test_losses.append(loss.data)
            if args.ngpu > 1:
                test_true.append(Y.data)
            else:
                test_true.append(Y.data)
            keys.extend(key)
            if pred.shape[1]==2:
                pred = torch.softmax(pred,dim = -1)[:,1]
            pred = pred if args.loss_fn == 'auc_loss' else pred
            test_pred.append(pred.data) if args.ngpu > 1 else test_pred.append(pred.data)

        # gather ngpu result to single tensor
        if args.ngpu > 1:
            test_true = distributed_concat(torch.concat(test_true, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
            test_pred = distributed_concat(torch.concat(test_pred, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
        
        else:
            test_true = torch.concat(test_true, dim=0).cpu().numpy()
            test_pred = torch.concat(test_pred, dim=0).cpu().numpy()
    return test_losses,test_true,test_pred,keys
def testAndPrint(model,loader,loss_fn,args,test_sampler):
    model.eval()
    df = pd.read_csv('sort_results.csv')
    results_df = pd.DataFrame(columns=['key', 'pred'])
    with torch.no_grad():
        test_losses,test_true,test_pred,keys = [], [],[],[]
        for i_batch, (g,full_g,Y,key) in tqdm.tqdm(enumerate(loader),total = len(loader)):
            # print(key)
            model.zero_grad()
            g = g.to(args.device,non_blocking=True)
            full_g = full_g.to(args.device,non_blocking=True)
            Y = Y.to(args.device,non_blocking=True)
            Y = Y.unsqueeze(-1)
            pred = model(g,full_g)
            loss = loss_fn(pred ,Y) 
            for k, p in zip(key, pred):
                df.loc[df['Name'] == k, 'RNAEquiScore'] = p.item()
            for k, p in zip(key, Y):
                df.loc[df['Name'] == k, 'True'] = p.item()
            # 将key和pred成对填入DataFrame
            batch_results = pd.DataFrame({'key': key, 'pred': pred.cpu().detach().numpy().flatten()})
            results_df = pd.concat([results_df, batch_results], ignore_index=True)
            if args.ngpu > 1:
                dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
                loss /= float(dist.get_world_size()) # get all loss value 
            # collect loss, true label and predicted label
            test_losses.append(loss.data)
            if args.ngpu > 1:
                test_true.append(Y.data)
            else:
                test_true.append(Y.data)
            keys.extend(key)
            if pred.shape[1]==2:
                pred = torch.softmax(pred,dim = -1)[:,1]
            pred = pred if args.loss_fn == 'auc_loss' else pred
            test_pred.append(pred.data) if args.ngpu > 1 else test_pred.append(pred.data)

        # gather ngpu result to single tensor
        if args.ngpu > 1:
            test_true = distributed_concat(torch.concat(test_true, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
            test_pred = distributed_concat(torch.concat(test_pred, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
        
        else:
            test_true = torch.concat(test_true, dim=0).cpu().numpy()
            test_pred = torch.concat(test_pred, dim=0).cpu().numpy()
        # df.to_csv('sort_results_with_rna_equi_score.csv', index=False)
        results_df.to_csv('VS_rank.csv', index=False)
    return test_losses,test_true,test_pred,keys
def testAndPrint_aff(model,loader,loss_fn,args,test_sampler):

    # g.ndata['type'] = node_types_tensor  # rna :1 or ligand: 2 or virtual: 0
    # g.ndata['bases'] = node_bases_tensor #base type in main_base_to_int;  ligand is 5 
    # g.ndata['base_idx'] = node_base_idx_tensor #base_idx   liake G_1;A_2
    # g.ndata['atom_idx'] = node_atom_idx_tensor #atom_idx 
    # g.ndata['atom_type'] = node_atom_type_tensor #atom_type
    # atom_type = GetNum(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H','other'])
    model.eval()
    with torch.no_grad():
        test_losses,test_true,test_pred,keys = [], [],[],[]
        sum_IFP = 0
        sum_noIFP = 0
        count_IFP = 0
        count_noIFP = 0
        sum_virual = 0
        count_virtual = 0
        sum_novirual = 0
        count_novirtual = 0
        for i_batch, (g,full_g,Y,key) in tqdm.tqdm(enumerate(loader),total = len(loader)):
            
            model.zero_grad()
            g = g.to(args.device,non_blocking=True)
            full_g = full_g.to(args.device,non_blocking=True)
            Y = Y.to(args.device,non_blocking=True)
            Y = Y.unsqueeze(-1)
            pred,score_matrix = model(g,full_g,getScore=True)
            loss = loss_fn(pred ,Y) 

            if args.ngpu > 1:
                dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
                loss /= float(dist.get_world_size()) # get all loss value 
            # collect loss, true label and predicted label
            test_losses.append(loss.data)
            if args.ngpu > 1:
                test_true.append(Y.data)
            else:
                test_true.append(Y.data)
            keys.extend(key)
            if pred.shape[1]==2:
                pred = torch.softmax(pred,dim = -1)[:,1]
            pred = pred if args.loss_fn == 'auc_loss' else pred
            test_pred.append(pred.data) if args.ngpu > 1 else test_pred.append(pred.data)
            if args.draw:
                # 节点属性提取
                num_nodes = full_g.num_nodes()
                src, dst = full_g.edges() 
                mean_scores = score_matrix.sum(dim=1).squeeze(-1).cpu()  # Shape: (num_edges,)
                # print(g.ndata)
                node_types = g.ndata['type'].cpu().numpy()  # 节点类型
                node_bases = g.ndata['bases'].cpu().numpy()  # 节点碱基类型
                node_base_idx = g.ndata['base_idx'].cpu().numpy()  # 碱基索引
                node_atom_types = g.ndata['atom_type'].cpu().numpy()  # 原子类型
                edge_types = full_g.edata['type']
                # 打印或存储边的详细信息
                edge_info = []
                for i in range(len(src)):
                    s, d = src[i].item(), dst[i].item()
                    edge_info.append({
                        "source": s,
                        "dest": d,
                        "weight": mean_scores[i].item(),
                        "src_type": node_types[s],
                        "src_base_idx":node_base_idx[s],
                        "dst_type": node_types[d],
                        "dst_base_idx":node_base_idx[d],
                        "src_base": node_bases[s],
                        "dst_base": node_bases[d],
                        "src_atom": node_atom_types[s],
                        "dst_atom": node_atom_types[d],
                        "edge_type": edge_types[i]
                    })
                # 可选：存储为文件，供后续分析
                import pandas as pd
                edge_df = pd.DataFrame(edge_info)
                edge_df.to_csv(f"data/graph/{key[0]}_x.csv", index=False)
                import csv
                import os

                # 定义原子类型和边类型名称
                atom_categories = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H', 'other']
                edge_type_names = {0: 'edge_type_0', 1: 'edge_type_1'}  # 边类型名称映射

                # 初始化统计字典（节点类别 + 边类型）
                stats = {category: {'sum': 0.0, 'count': 0} for category in ['virtual'] + atom_categories}
                edge_type_stats = {et: {'sum': 0.0, 'count': 0} for et in edge_type_names}  # 边类型统计

                for edge in edge_info:
                    src_type = edge['src_type']
                    dst_type = edge['dst_type']
                    weight = edge['weight']
                    et = edge['edge_type'].item()  # 边类型（0或1）
                    
                    # 统计节点类别（virtual或原子类型）
                    if src_type == 0 or dst_type == 0:
                        stats['virtual']['sum'] += weight
                        stats['virtual']['count'] += 1
                        
                        sum_virual += weight
                        count_virtual += 1
                    else:
                        sum_novirual += weight
                        count_novirtual += 1
                        src_atom = atom_categories[edge['src_atom']]
                        dst_atom = atom_categories[edge['dst_atom']]
                        stats[src_atom]['sum'] += weight
                        stats[src_atom]['count'] += 1
                        stats[dst_atom]['sum'] += weight
                        stats[dst_atom]['count'] += 1
                    
                    # 统计边类型（无论是否连接到virtual节点）
                    edge_type_stats[et]['sum'] += weight
                    edge_type_stats[et]['count'] += 1

                # 计算平均值（节点类别）
                averages = {}
                for category in ['virtual'] + atom_categories:
                    total = stats[category]['sum']
                    count = stats[category]['count']
                    averages[category] = total / count if count != 0 else 0.0

                # 计算平均值（边类型）
                for et in edge_type_names:
                    total = edge_type_stats[et]['sum']
                    count = edge_type_stats[et]['count']
                    averages[edge_type_names[et]] = total / count if count != 0 else 0.0
                sum_IFP += edge_type_stats[0]['sum']
                sum_noIFP = edge_type_stats[1]['sum']
                count_IFP = edge_type_stats[0]['count']
                count_noIFP = edge_type_stats[1]['count']

                # 确保输出目录存在
                output_dir = 'data/graph'
                os.makedirs(output_dir, exist_ok=True)

                # 生成CSV文件（包含节点类别和边类型）
                output_path = os.path.join(output_dir, f'{key[0]}.csv')
                ordered_categories = ['virtual'] + atom_categories + list(edge_type_names.values())

                with open(output_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['category', 'average_weight'])
                    for cat in ordered_categories:
                        writer.writerow([cat, averages.get(cat, 0.0)])

                # print(f"结果已保存至 {output_path}")
               

                # Create a zero-initialized adjacency matrix
                adj_matrix = torch.zeros((num_nodes, num_nodes))

                # Populate the adjacency matrix with mean scores
                adj_matrix[src, dst] = mean_scores  # Use src, dst to map scores to matrix

                # To handle undirected graphs, optionally add scores for reverse edges
                # adj_matrix[dst, src] = mean_scores

                # Visualize as a heatmap
                plt.figure(figsize=(10, 8))
                plt.imshow(adj_matrix.numpy(), cmap='viridis', interpolation='none')
                plt.colorbar(label="Edge Weight (Mean Score)")
                plt.title("Multi-Head Attention Heatmap")
                plt.xlabel("Node Index")
                plt.ylabel("Node Index")
                plt.savefig(f"fig/score_{key}.png")
                plt.close()
        if args.draw:
            print(f'sum_virtual:{sum_virual};count_virtual:{count_virtual};sum_novirtual:{sum_novirual};count_novirtual:{count_novirtual};sum_IFP:{sum_IFP};count_IFP:{count_IFP};sum_noIFP:{sum_noIFP};count_noIFP:{count_noIFP}')
            print(f'{sum_virual/count_virtual}      {sum_novirual/count_novirtual}       {sum_IFP/count_IFP}          {sum_noIFP/count_noIFP}')
        # gather ngpu result to single tensor
        if args.ngpu > 1:
            test_true = distributed_concat(torch.concat(test_true, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
            test_pred = distributed_concat(torch.concat(test_pred, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
        
        else:
            test_true = torch.concat(test_true, dim=0).cpu().numpy()
            test_pred = torch.concat(test_pred, dim=0).cpu().numpy()
    return test_losses,test_true,test_pred,keys
def testAndPrint_aff_equi(model,loader,loss_fn,args,test_sampler):
    model.eval()
    with torch.no_grad():
        test_losses,test_true,test_pred,keys = [], [],[],[]
        for i_batch, (g,full_g,Y,key) in tqdm.tqdm(enumerate(loader),total = len(loader)):
            batch = g.batch_num_nodes()
            model.zero_grad()
            g = g.to(args.device,non_blocking=True)
            # batch = g.batch_num_nodes()
            full_g = full_g.to(args.device,non_blocking=True)
            Y = Y.to(args.device,non_blocking=True)
            Y = Y.unsqueeze(-1)
            batch = torch.arange(len(Y)).repeat_interleave(batch).to(args.device)
            
            pred = model(g,batch)
            loss = loss_fn(pred ,Y) 

            if args.ngpu > 1:
                dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
                loss /= float(dist.get_world_size()) # get all loss value 
            # collect loss, true label and predicted label
            test_losses.append(loss.data)
            if args.ngpu > 1:
                test_true.append(Y.data)
            else:
                test_true.append(Y.data)
            keys.extend(key)
            if pred.shape[1]==2:
                pred = torch.softmax(pred,dim = -1)[:,1]
            pred = pred if args.loss_fn == 'auc_loss' else pred
            test_pred.append(pred.data) if args.ngpu > 1 else test_pred.append(pred.data)

        # gather ngpu result to single tensor
        if args.ngpu > 1:
            test_true = distributed_concat(torch.concat(test_true, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
            test_pred = distributed_concat(torch.concat(test_pred, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
        
        else:
            test_true = torch.concat(test_true, dim=0).cpu().numpy()
            test_pred = torch.concat(test_pred, dim=0).cpu().numpy()
    return test_losses,test_true,test_pred,keys
def testAndPrint_equi(model,loader,loss_fn,args,test_sampler):
    model.eval()
    
    with torch.no_grad():
        test_losses,test_true,test_pred,keys = [], [],[],[]
        for i_batch, (g,Y,key,batch) in tqdm.tqdm(enumerate(loader),total = len(loader)):
            batch = g.batch_num_nodes()
            model.zero_grad()
            g = g.to(args.device,non_blocking=True)
            # full_g = full_g.to(args.device,non_blocking=True)
            Y = Y.to(args.device,non_blocking=True)
            Y = Y.unsqueeze(-1)
            # 将 batch 转换为适合 radius_graph 的格式
            # print(batch)
            batch = torch.arange(len(Y)).repeat_interleave(batch).to(args.device)
            
            # logits = model(g, batch)
            pred = model(g,batch)
            loss = loss_fn(pred ,Y) 
            # 假设key是一个包含Name的列表
            
            if args.ngpu > 1:
                dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
                loss /= float(dist.get_world_size()) # get all loss value 
            # collect loss, true label and predicted label
            test_losses.append(loss.data)
            if args.ngpu > 1:
                test_true.append(Y.data)
            else:
                test_true.append(Y.data)
            keys.extend(key)
            if pred.shape[1]==2:
                pred = torch.softmax(pred,dim = -1)[:,1]
            pred = pred if args.loss_fn == 'auc_loss' else pred
            test_pred.append(pred.data) if args.ngpu > 1 else test_pred.append(pred.data)
            
        # gather ngpu result to single tensor
        if args.ngpu > 1:
            test_true = distributed_concat(torch.concat(test_true, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
            test_pred = distributed_concat(torch.concat(test_pred, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
        
        else:
            test_true = torch.concat(test_true, dim=0).cpu().numpy()
            test_pred = torch.concat(test_pred, dim=0).cpu().numpy()
        results = pd.DataFrame({
            'key': keys,
            'true': test_true.flatten(),
            'pred': test_pred.flatten()
        })
        results.to_csv('test_results.csv', index=False)
        print("Results saved to test_results.csv")
        
    return test_losses,test_true,test_pred,keys
import copy
import tqdm
def train(model,args,optimizer,loss_fn,train_dataloader,scheduler):
    # collect losses of each iteration
    train_losses = [] 
    model.train()

    for i_batch, (g,full_g,Y,key) in tqdm.tqdm(enumerate(train_dataloader),total = len(train_dataloader)):
        g = g.to(args.device,non_blocking=True)
        full_g = full_g.to(args.device,non_blocking=True)

        Y = Y.to(args.device,non_blocking=True)
        if args.lap_pos_enc:
            batch_lap_pos_enc = g.ndata['lap_pos_enc']
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(args.device,non_blocking=True)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            g.ndata['lap_pos_enc'] = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        # print("flag")
       
        logits = model(g,full_g)
        Y = Y.unsqueeze(-1)
        # print(logits)
        # print(Y.shape)
        loss = loss_fn(logits, Y)
        train_losses.append(loss.item())
        # print(type(loss.item()))
        loss = loss/args.grad_sum
        loss.backward()
        if (i_batch + 1) % args.grad_sum == 0  or i_batch == len(train_dataloader) - 1:
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()

        if args.ngpu > 1:
            dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
            loss /= float(dist.get_world_size()) # get all loss value 
        loss = loss.data*args.grad_sum 
        if args.lr_decay:
            scheduler.step()
    return model,train_losses,optimizer,scheduler
def pcc_loss(y_pred, y_true):
    # 计算皮尔逊相关系数损失
    y_pred_mean = torch.mean(y_pred)
    y_true_mean = torch.mean(y_true)
    
    cov = torch.mean((y_pred - y_pred_mean) * (y_true - y_true_mean))
    std_pred = torch.std(y_pred)
    std_true = torch.std(y_true)
    
    pcc = cov / (std_pred * std_true)
    return pcc  

def spcc_loss(y_pred, y_true):
    # 计算斯皮尔曼相关系数损失
    pred_rank = torch.argsort(torch.argsort(y_pred))
    true_rank = torch.argsort(torch.argsort(y_true))
    
    n = y_pred.size(0)
    diff_rank = pred_rank - true_rank
    spcc = 1 - (6 * torch.sum(diff_rank ** 2)) / (n * (n ** 2 - 1))
    return spcc  
def train_aff(model,args,optimizer,loss_fn,train_dataloader,scheduler):
    # collect losses of each iteration
    train_losses = [] 
    model.train()

    for i_batch, (g,full_g,Y,key) in tqdm.tqdm(enumerate(train_dataloader),total = len(train_dataloader)):
        optimizer.zero_grad()
        model.zero_grad()
        g = g.to(args.device,non_blocking=True)
        full_g = full_g.to(args.device,non_blocking=True)

        Y = Y.to(args.device,non_blocking=True)
        if args.lap_pos_enc:
            batch_lap_pos_enc = g.ndata['lap_pos_enc']
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(args.device,non_blocking=True)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            g.ndata['lap_pos_enc'] = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        # print("flag")
       
        logits, score_matrix = model(g,full_g,getScore=True)
        Y = Y.unsqueeze(-1)
        # print(logits)
        # print(Y.shape)
        mse_loss = loss_fn(logits, Y)
        # PCC损失
        pcc_loss_value = pcc_loss(logits, Y)
        
        # SPCC损失
        spcc_loss_value = spcc_loss(logits, Y)

        # 组合损失 (可以根据需要调整权重)
        loss = args.mse_weight * mse_loss - args.pcc_weight * pcc_loss_value - args.spcc_weight * spcc_loss_value
        
        train_losses.append(loss.item())
        # print(type(loss.item()))
        loss = loss/args.grad_sum
        loss.backward()
        if (i_batch + 1) % args.grad_sum == 0  or i_batch == len(train_dataloader) - 1:
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()

        if args.ngpu > 1:
            dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
            loss /= float(dist.get_world_size()) # get all loss value 
        loss = loss.data*args.grad_sum 
        if args.lr_decay:
            scheduler.step()
    return model,train_losses,optimizer,scheduler
def contrastive_loss(pos_sim, neg_sim, temperature=0.07):
    # 将相似度缩放并应用softmax
    pos_sim = pos_sim / temperature
    neg_sim = neg_sim / temperature
    
    # 最大化正样本的相似度，最小化负样本的相似度
    pos_loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
    
    # 取平均值作为最终损失
    return torch.mean(pos_loss)
def train_contrastive(model,args,optimizer,loss_fn,train_dataloader,scheduler):
    # collect losses of each iteration
    train_losses = [] 
    loss_logits = []
    loss_conts = []
    # lossLR=conLoss(args.batch_size).to(args.device)
    model.train()
    # alpha=1.0
    for i_batch, (g_batch,full_g_batch,Y) in tqdm.tqdm(enumerate(train_dataloader),total = len(train_dataloader)):
        optimizer.zero_grad()
        model.zero_grad()
#         g_pos1 = g_pos1.to(args.device,non_blocking=True)
#         full_g_pos1 = full_g_pos1.to(args.device,non_blocking=True)
#         g_pos2 = g_pos2.to(args.device,non_blocking=True)
#         full_g_pos2 = full_g_pos2.to(args.device,non_blocking=True)
        
#         g_neg1 = g_neg1.to(args.device,non_blocking=True)
#         full_g_neg1 = full_g_neg1.to(args.device,non_blocking=True)
#         g_neg2 = g_neg2.to(args.device,non_blocking=True)
#         full_g_neg2 = full_g_neg2.to(args.device,non_blocking=True)
        # 合并样本
        # g_batch = dgl.batch([g_pos1, g_pos2, g_neg1, g_neg2]).to(args.device, non_blocking=True)
        # full_g_batch = dgl.batch([full_g_pos1, full_g_pos2, full_g_neg1, full_g_neg2]).to(args.device, non_blocking=True)
        g_batch = g_batch.to(args.device, non_blocking=True)
        full_g_batch = full_g_batch.to(args.device, non_blocking=True)
        # 模型前向传播，得到所有样本的嵌入
        embeddings, logits = model(g_batch, full_g_batch,contrastive = True)
        Y = Y.to(args.device,non_blocking=True)
        Y = Y.unsqueeze(-1)
        loss_logit = loss_fn(logits, Y)
        # PCC损失
        pcc_loss_value = pcc_loss(logits, Y)
        
        # SPCC损失
        spcc_loss_value = spcc_loss(logits, Y)

        
        # 拆分输出
        batch_size = args.batch_size
        pos1, pos2 = embeddings[0:batch_size], embeddings[batch_size:2*batch_size]
        neg1, neg2 = embeddings[2*batch_size:3*batch_size], embeddings[3*batch_size:]
        
        # print("flag")
        # pos1,pos2 = model(g_pos1,full_g_pos1),model(g_pos2,full_g_pos2)
        # neg1,neg2 = model(g_neg1,full_g_neg1),model(g_neg2,full_g_neg2)
        
        # print(logits)
        # print(Y.shape)
        pos_sim = F.cosine_similarity(pos1,pos2)
        neg_sim = F.cosine_similarity(pos1,neg1)
        loss_cont = contrastive_loss(pos_sim, neg_sim)
        # 计算对比学习损失
        # loss =  loss_cont + loss_logit
        # if loss_cont<0.1:
        alpha=0.4
        
        loss1 = alpha * loss_cont + (1 - alpha) * loss_logit
        # 组合损失 (可以根据需要调整权重)
        loss = args.mse_weight * loss1 - args.pcc_weight * pcc_loss_value - args.spcc_weight * spcc_loss_value
        # 反向传播并更新模型参数
        loss.backward()
        optimizer.step()

        # 打印损失
        train_losses.append(loss.item())
        loss_conts.append(loss_cont.item())
        loss_logits.append(loss_logit.item())
        if args.lr_decay:
            scheduler.step()
        torch.cuda.empty_cache()
    return model,train_losses,loss_conts,loss_logits,optimizer,scheduler
def train_aff_equi(model,args,optimizer,loss_fn,train_dataloader,scheduler):
    # collect losses of each iteration
    train_losses = [] 
    model.train()

    for i_batch, (g,full_g,Y,key) in tqdm.tqdm(enumerate(train_dataloader),total = len(train_dataloader)):
        batch = g.batch_num_nodes()
        g = g.to(args.device,non_blocking=True)
        # full_g = full_g.to(args.device,non_blocking=True)
        
        Y = Y.to(args.device,non_blocking=True)
        if args.lap_pos_enc:
            batch_lap_pos_enc = g.ndata['lap_pos_enc']
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(args.device,non_blocking=True)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            g.ndata['lap_pos_enc'] = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        # print("flag")
       
        batch = torch.arange(len(Y)).repeat_interleave(batch).to(args.device)
        
        logits = model(g, batch)
        Y = Y.unsqueeze(-1)
        # print(logits)
        # print(Y.shape)
        loss = loss_fn(logits, Y)
        train_losses.append(loss.item())
        # print(type(loss.item()))
        loss = loss/args.grad_sum
        loss.backward()
        if (i_batch + 1) % args.grad_sum == 0  or i_batch == len(train_dataloader) - 1:
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()

        if args.ngpu > 1:
            dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
            loss /= float(dist.get_world_size()) # get all loss value 
        loss = loss.data*args.grad_sum 
        if args.lr_decay:
            scheduler.step()
    return model,train_losses,optimizer,scheduler
def train_equi(model,args,optimizer,loss_fn,train_dataloader,scheduler):
    # collect losses of each iteration
    train_losses = [] 
    model.train()

    for i_batch, (g,Y,key,batch) in tqdm.tqdm(enumerate(train_dataloader),total = len(train_dataloader)):
        # continue
        batch = g.batch_num_nodes()
        g = g.to(args.device,non_blocking=True)
        # full_g = full_g.to(args.device,non_blocking=True)
        # batch = g.size(0)
        Y = Y.to(args.device,non_blocking=True)
        if args.lap_pos_enc:
            batch_lap_pos_enc = g.ndata['lap_pos_enc']
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(args.device,non_blocking=True)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            g.ndata['lap_pos_enc'] = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        # print("flag")
        # print(batch.numel())# 获取批图中的批次信息
        
        
        # 将 batch 转换为适合 radius_graph 的格式
        batch = torch.arange(len(Y)).repeat_interleave(batch).to(args.device)
        
        logits = model(g, batch)
        Y = Y.unsqueeze(-1)
        # print(logits)
        # print(Y.shape)
        loss = loss_fn(logits, Y)
        train_losses.append(loss.item())
        # print(type(loss.item()))
        loss = loss/args.grad_sum
        loss.backward()
        if (i_batch + 1) % args.grad_sum == 0  or i_batch == len(train_dataloader) - 1:
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()

        if args.ngpu > 1:
            dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
            loss /= float(dist.get_world_size()) # get all loss value 
        loss = loss.data*args.grad_sum 
        if args.lr_decay:
            scheduler.step()
    return model,train_losses,optimizer,scheduler
def getToyKey(train_keys):

    """get toy dataset for test"""

    train_keys_toy_d = []
    train_keys_toy_a = []
    
    max_all = 600
    for key in train_keys:
        if '_active_' in key:
            train_keys_toy_a.append(key)
        if '_active_' not in key:
            train_keys_toy_d.append(key)

    if len(train_keys_toy_a) == 0 or len(train_keys_toy_d) == 0:
        return None

    return train_keys_toy_a[:300] + train_keys_toy_d[:(max_all-300)]
def getTestedPro(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            lines = []
            for line in f.readlines():
                if 'actions' in line:
                    lines.append(line)
        lines = [line.split('\t')[0] for line in lines]
        return lines
    else:
        return []
def getEF(model,args,test_path,save_path,debug,batch_size,loss_fn,rates = 0.01,flag = '',prot_split_flag = '_'):
        """calculate EF of test dataset, since dataset have 102/81 proteins ,so we need to calculate EF of each protein one by one!"""
        save_file = save_path + '/EF_test' + flag
        tested_pros = getTestedPro(save_file)
        test_keys = [key for key in os.listdir(test_path) if '.' not in key]
        pros = defaultdict(list)
        for key in test_keys:
            key_split = key.split(prot_split_flag)
 
            if '_active' in key:
                pros[key_split[0]].insert(0,os.path.join(test_path ,key))
            else:
                ''' all positive label sample will be place in head of list'''
                pros[key_split[0]].append(os.path.join(test_path ,key))


        EFs = []
        st = time.time()
        if type(rates) is not list:
                rates = list([rates])
        rate_str = ''
        for rate in rates:
            rate_str += str(rate)+ '\t'
        for pro in pros.keys():
            try :
                if pro in tested_pros:
                    if args.ngpu >= 1:
                        dist.barrier()
                    print('this pro :  %s  is tested'%pro)
                    continue
                test_keys_pro = pros[pro]
                if len(test_keys_pro) == 0:
                    if args.ngpu >= 1:
                        dist.barrier()
                    continue
                test_dataset = ESDataset(test_keys_pro,args, test_path,debug)
                val_sampler = SequentialDistributedSampler(test_dataset,args.batch_size) if args.ngpu >= 1 else None
                test_dataloader = DataLoaderX(test_dataset, batch_size = batch_size, \
                shuffle=False, num_workers = 8, collate_fn=test_dataset.collate,pin_memory = True,sampler = val_sampler)

                test_losses,test_true,test_pred = evaluator(model,test_dataloader,loss_fn,args,val_sampler)

                if args.ngpu >= 1:
                    dist.barrier()
                if args.local_rank == 0:
                    test_auroc,BEDROC,test_adjust_logauroc,test_auprc,test_balanced_acc,test_acc,test_precision,test_sensitity,test_specifity,test_f1 = get_metrics(test_true,test_pred)
                    test_losses = torch.mean(torch.tensor(test_losses,dtype=torch.float)).data.cpu().numpy()
                    Y_sum = 0
                    for key in test_keys_pro:
                        key_split = key.split('_')
                        if '_active' in key:
                            Y_sum += 1
                    actions = int(Y_sum)
                    action_rate = actions/len(test_keys_pro)

                    EF = []
                    hits_list = []
                    for rate in rates:
                        ''' cal different rates of EF'''
                        find_limit = int(len(test_keys_pro)*rate)
                        _,indices = torch.sort(torch.tensor(test_pred),descending = True)
                        hits = torch.sum(indices[:find_limit] < actions)
                        EF.append((hits/find_limit)/action_rate)
                        hits_list.append(hits)

                    
                    EF_str = '['
                    hits_str = '['
                    for ef,hits in zip(EF,hits_list):
                        EF_str += '%.3f'%ef+'\t'
                        hits_str += ' %d '%hits
                    EF_str += ']'
                    hits_str += ']'
                    end = time.time()
                    with open(save_file,'a') as f:
                        f.write(pro+ '\t'+'actions: '+str(actions)+ '\t' + 'actions_rate: '+str(action_rate)+ '\t' + 'hits: '+ hits_str +'\t'+'loss:' + str(test_losses)+'\n'\
                            +'EF:'+rate_str+ '\t'+'test_auroc'+ '\t' + 'BEDROC' + '\t'+'test_adjust_logauroc'+ '\t'+'test_auprc'+ '\t'+'test_balanced_acc'+ '\t'+'test_acc'+ '\t'+'test_precision'+ '\t'+'test_sensitity'+ '\t'+'test_specifity'+ '\t'+'test_f1' +'\t' +'time'+ '\n')
                        f.write( EF_str + '\t'+str(test_auroc)+ '\t' + str(BEDROC) + '\t'+str(test_adjust_logauroc)+ '\t'+str(test_auprc)+ '\t'+str(test_balanced_acc)+ '\t'+str(test_acc)+ '\t'+str(test_precision)+ '\t'+str(test_sensitity)+ '\t'+str(test_specifity)+ '\t'+str(test_f1) +'\t'+ str(end-st)+ '\n')
                        f.close()
                    EFs.append(EF)
            except:
                print(pro,':skip for some bug')
                if args.ngpu >= 1:
                    dist.barrier()
                continue
            if args.ngpu >= 1:
                dist.barrier()
        if args.local_rank == 0:
            EFs = list(np.sum(np.array(EFs),axis=0)/len(EFs))
            EFs_str = '['
            for ef in EFs:
                EFs_str += str(ef)+'\t'
            EFs_str += ']'
            args_dict = vars(args)
            with open(save_file,'a') as f:
                    f.write( 'average EF for different EF_rate:' + EFs_str +'\n')
                    for item in args_dict.keys():
                        f.write(item + ' : '+str(args_dict[item]) + '\n')
                    f.close()
        if args.ngpu >= 1:
            # Keeping processes in sync
            dist.barrier()
def getNumPose(test_keys,nums = 5):

    """get the first nums pose for each ligand to prediction"""

    ligands = defaultdict(list)
    for key in test_keys:
        key_split = key.split('_')
        ligand_name = '_'.join(key_split[-2].split('-')[:-1])
        ligands[ligand_name].append(key)
    result = []
    for ligand_name in ligands.keys():
        ligands[ligand_name].sort(key = lambda x : int(x.split('_')[-2].split('-')[-1]),reverse=False)
        result += ligands[ligand_name][:nums]
    return result
def getIdxPose(test_keys,idx = 0):
    """"get the idx pose for each ligand to prediction"""
    ligands = defaultdict(list)


    for key in test_keys:
        key_split = key.split('_')
        ligand_name = '_'.join(key_split[-2].split('-')[:-1])
        ligands[ligand_name].append(key)
    result = []
    for ligand_name in ligands.keys():

        ligands[ligand_name].sort(key = lambda x : int(x.split('_')[-2].split('-')[-1]),reverse=False)
        if idx < len(ligands[ligand_name]):
            result.append(ligands[ligand_name][idx]) 
        else:
            result.append(ligands[ligand_name][-1]) 
    return result
def getEFMultiPose(model,args,test_path,save_path,debug,batch_size,loss_fn,rates = 0.01,flag = '',pose_num = 5,idx_style = False):
        """calulate EF for multi pose complex"""
        save_file = save_path + '/EF_test_multi_pose' + '_{}_'.format(pose_num) + flag
        test_keys = os.listdir(test_path)
        # for multi pose complex, get pose_num poses to cal EF
        tested_pros = getTestedPro(save_file)
        if idx_style:
            test_keys = getIdxPose(test_keys,idx = pose_num)
        else:
            test_keys = getNumPose(test_keys,nums = pose_num) 
 
        pros = defaultdict(list)
        for key in test_keys:
            key_split = key.split('_')
            pros[key_split[0]].append(os.path.join(test_path , key))
        EFs = []
        st = time.time()
        if type(rates) is not list:
                rates = list([rates])
        rate_str = ''
        for rate in rates:
            rate_str += str(rate)+ '\t'
        for pro in pros.keys():
            try :
                if pro in tested_pros:
                    if args.ngpu >= 1:
                        dist.barrier()
                    print('this pro :  %s  is tested'%pro)
                    continue
                test_keys_pro = pros[pro]
                if test_keys_pro is None:
                    if args.ngpu >= 1:
                        dist.barrier()
                    continue
                print('protein keys num ',len(test_keys_pro))

                test_dataset = ESDataset(test_keys_pro,args, test_path,debug)
                val_sampler = SequentialDistributedSampler(test_dataset,args.batch_size) if args.ngpu >= 1 else None
                test_dataloader = DataLoaderX(test_dataset, batch_size = batch_size, \
                shuffle=False, num_workers = 8, collate_fn=test_dataset.collate,pin_memory = True,sampler = val_sampler)
                test_losses,test_true,test_pred = evaluator(model,test_dataloader,loss_fn,args,val_sampler)

                if args.ngpu >= 1:
                    dist.barrier()
                if args.local_rank == 0:
 
                    test_auroc,BEDROC,test_adjust_logauroc,test_auprc,test_balanced_acc,test_acc,test_precision,test_sensitity,test_specifity,test_f1 = get_metrics(test_true,test_pred)
                    test_losses = torch.mean(torch.tensor(test_losses,dtype=torch.float)).data.cpu().numpy()
                    Y_sum = 0
                    # multi pose 
                    # get max logits for every ligand
                    key_logits = defaultdict(list)
                    for pred,key in zip(test_pred,test_keys_pro):
                        new_key = '_'.join(key.split('/')[-1].split('_')[:-2] + key.split('/')[-1].split('_')[-2].split('-')[:-1])
                        key_logits[new_key].append(pred)
                    new_keys = list(key_logits.keys())
                    max_pose_logits = [max(logits) for logits in  list(key_logits.values())]

                    test_keys_pro = []
                    test_pred = []
                    for key,logit in zip(new_keys,max_pose_logits):
                        key_split = key.split('_') 
                        if 'actives' in key_split:
                            test_keys_pro.insert(0,key)
                            test_pred.insert(0,logit)
                            Y_sum += 1
                        else:
                            ''' all positive label sample will be place in head of list'''
                            test_keys_pro.append(key)
                            test_pred.append(logit)

                    actions = int(Y_sum)
                    action_rate = actions/len(test_keys_pro)
                    
                    EF = []
                    hits_list = []
                    for rate in rates:
                        find_limit = int(len(test_keys_pro)*rate)
                        _,indices = torch.sort(torch.tensor(test_pred),descending = True)
                        hits = torch.sum(indices[:find_limit] < actions)
                        EF.append((hits/find_limit)/action_rate)
                        hits_list.append(hits)
                    
                    EF_str = '['
                    hits_str = '['
                    for ef,hits in zip(EF,hits_list):
                        EF_str += '%.3f'%ef+'\t'
                        hits_str += ' %d '%hits
                    EF_str += ']'
                    hits_str += ']'
                    end = time.time()
                    with open(save_file,'a') as f:
                        f.write(pro+ '\t'+'actions: '+str(actions)+ '\t' + 'actions_rate: '+str(action_rate)+ '\t' + 'hits: '+ hits_str +'\t'+'loss:' + str(test_losses)+'\n'\
                            +'EF:'+rate_str+ '\t'+'test_auroc'+ '\t' + 'BEDROC' + '\t'+'test_adjust_logauroc'+ '\t'+'test_auprc'+ '\t'+'test_balanced_acc'+ '\t'+'test_acc'+ '\t'+'test_precision'+ '\t'+'test_sensitity'+ '\t'+'test_specifity'+ '\t'+'test_f1' +'\t' +'time'+ '\n')
                        f.write( EF_str + '\t'+str(test_auroc)+ '\t' + str(BEDROC) + '\t'+str(test_adjust_logauroc)+ '\t'+str(test_auprc)+ '\t'+str(test_balanced_acc)+ '\t'+str(test_acc)+ '\t'+str(test_precision)+ '\t'+str(test_sensitity)+ '\t'+str(test_specifity)+ '\t'+str(test_f1) +'\t'+ str(end-st)+ '\n')
                        f.close()
                    EFs.append(EF)
            except:
                print(pro,':skip for some bug')
                if args.ngpu >= 1:
                    dist.barrier()
                continue
            if args.ngpu >= 1:
                dist.barrier()
        if args.local_rank == 0:
            EFs = list(np.sum(np.array(EFs),axis=0)/len(EFs))
            EFs_str = '['
            for ef in EFs:
                EFs_str += str(ef)+'\t'
            EFs_str += ']'
            args_dict = vars(args)
            with open(save_file,'a') as f:
                    f.write( 'average EF for different EF_rate:' + EFs_str +'\n')
                    for item in args_dict.keys():
                        f.write(item + ' : '+str(args_dict[item]) + '\n')
                    f.close()
        if args.ngpu >= 1:
            dist.barrier()
def getEF_from_MSE(model,args,test_path,save_path,device,debug,batch_size,A2_limit,loss_fn,rates = 0.01):
        """cal EF for regression model if you want to training a regression model, you can use this function to cal EF"""
        
        save_file = save_path + '/EF_test'
        test_keys = [key for key in os.listdir(test_path) if '.' not in key]
        pros = defaultdict(list)
        for key in test_keys:
            key_split = key.split('_')
            if 'active' in key_split:
                pros[key_split[0]].insert(0,key)
            else:
                pros[key_split[0]].append(key)

        EFs = []
        st = time.time()
        if type(rates) is not list:
                rates = list([rates])
        rate_str = ''
        for rate in rates:
            rate_str += str(rate)+ '\t'
        for pro in pros.keys():
            try :

                test_keys_pro = pros[pro]
                if test_keys_pro is None:
                    continue
                test_dataset = ESDataset(test_keys_pro,args, test_path,debug)
                test_dataloader = DataLoader(test_dataset, batch_size = batch_size, \
                shuffle=False, num_workers = args.num_workers, collate_fn=test_dataset.collate)
                test_losses,test_true,test_pred = evaluator(model,test_dataloader,loss_fn,args)
                test_auroc,test_adjust_logauroc,test_auprc,test_balanced_acc,test_acc,test_precision,test_sensitity,test_specifity,test_f1 = get_metrics(test_true,test_pred)
                test_losses = np.mean(np.array(test_losses))
                # print(test_losses)
                Y_sum = 0
                for key in test_keys_pro:
                    key_split = key.split('_')
                    if 'active' in key_split:
                        Y_sum += 1
                actions = int(Y_sum)
                action_rate = actions/len(test_keys_pro)
                test_pred = np.concatenate(np.array(test_pred), 0)
                EF = []
                hits_list = []
                for rate in rates:
                    find_limit = int(len(test_keys_pro)*rate)
                    _,indices = torch.sort(torch.tensor(test_pred),descending = True)
                    hits = torch.sum(indices[:find_limit] < actions)
                    EF.append((hits/find_limit)/action_rate)
                    hits_list.append(hits)
                
                EF_str = '['
                hits_str = '['
                for ef,hits in zip(EF,hits_list):
                    EF_str += '%.3f'%ef+'\t'
                    hits_str += ' %d '%hits
                EF_str += ']'
                hits_str += ']'
                end = time.time()
                with open(save_file,'a') as f:
                    f.write(pro+ '\t'+'actions: '+str(actions)+ '\t' + 'actions_rate: '+str(action_rate)+ '\t' + 'hits: '+ hits_str +'\n'\
                        +'EF:'+rate_str)
                    f.write( EF_str)
                    f.close()
                EFs.append(EF)
            except:
                print(pro,':skip for some bug')
                continue
        EFs = list(np.sum(np.array(EFs),axis=0)/len(EFs))
        EFs_str = '['
        for ef in EFs:
            EFs_str += str(ef)+'\t'
        EFs_str += ']'
        args_dict = vars(args)
        with open(save_file,'a') as f:
                f.write( 'average EF for different EF_rate:' + EFs_str +'\n')
                for item in args_dict.keys():
                    f.write(item + ' : '+str(args_dict[item]) + '\n')
                f.close()
from collections import defaultdict
import numpy as np
import pickle

def get_train_val_keys(keys):
    train_keys = keys
    pro_dict = defaultdict(list)
    for key in train_keys:
        pro = key.split('_')[0]
        pro_dict[pro].append(key)
    pro_list = list(pro_dict.keys())
    indices = np.arange(len(pro_list))
    np.random.shuffle(indices)
    train_num = int(len(indices)*0.8)
    count = 0
    train_list = []
    val_list = []
    for i in indices:
        count +=1
        if count < train_num:
            train_list += pro_dict[pro_list[i]]
        else:
            val_list +=  pro_dict[pro_list[i]]
    return train_list,val_list
def get_dataloader(args,train_keys,val_keys,val_shuffle=False):
    """"
    docstring:
        get dataloader for train and validation
    input:
        train_keys: list of train keys
            train file paths
        val_keys: list of validation keys
            validation file paths

    output: dataloader for train and validation
        (train_dataloader,val_dataloader)
    """
    train_dataset = ESDataset(train_keys,args, args.data_path,args.debug)
    val_dataset = ESDataset(val_keys,args, args.data_path,args.debug)
   
    if args.sampler:

        num_train_chembl = len([0 for k in train_keys if '_active' in k])
        num_train_decoy = len([0 for k in train_keys if '_active' not in k])
        train_weights = [1/num_train_chembl if '_active' in k else 1/num_train_decoy for k in train_keys]
        train_sampler = DTISampler(train_weights, len(train_weights), replacement=True)                     
        train_dataloader = DataLoader(train_dataset, args.batch_size, \
            shuffle=False,num_workers = args.num_workers, collate_fn=train_dataset.collate,\
            sampler = train_sampler)
    else:
        train_dataloader = DataLoader(train_dataset, args.batch_size, \
            shuffle=True, num_workers = args.num_workers, collate_fn=train_dataset.collate)
    val_dataloader = DataLoader(val_dataset, args.batch_size, \
        shuffle=val_shuffle, num_workers = args.num_workers, collate_fn=val_dataset.collate)
    return train_dataloader,val_dataloader

def write_log_head(args,log_path,model,train_keys,val_keys):
    """a function to write the head of log file at the beginning of training"""
    args_dict = vars(args)
    with open(log_path,'w')as f:
        f.write(f'Number of train data: {len(train_keys)}' +'\n'+ f'Number of val data: {len(val_keys)}' + '\n')
        f.write(f'number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}' +'\n')
        for item in args_dict.keys():
            f.write(item + ' : '+str(args_dict[item]) + '\n')
        f.write('epoch'+'\t'+'train_loss'+'\t'+'val_loss'+'\t'+'test_loss' #'\t'+'train_auroc'+ '\t'+'train_adjust_logauroc'+ '\t'+'train_auprc'+ '\t'+'train_balanced_acc'+ '\t'+'train_acc'+ '\t'+'train_precision'+ '\t'+'train_sensitity'+ '\t'+'train_specifity'+ '\t'+'train_f1'+ '\t'\
        + '\t' + 'test_auroc'+ '\t' + 'BEDROC' + '\t'+'test_adjust_logauroc'+ '\t'+'test_auprc'+ '\t'+'test_balanced_acc'+ '\t'+'test_acc'+ '\t'+'test_precision'+ '\t'+'test_sensitity'+ '\t'+'test_specifity'+ '\t'+'test_f1' +'\t' +'time'+ '\n')
        f.close()
def save_model(model,optimizer,args,epoch,save_path,cv=0,mode = 'best'):
    """a function to save model"""
    best_name = save_path + f'/save_{mode}_model_{cv}'+'.pt'
    if args.debug:
        best_name = save_path + '/save_{}_model_debug'.format(mode)+'.pt'

    torch.save({'model':model.module.state_dict() if isinstance(model,nn.parallel.DistributedDataParallel) else model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'epoch':epoch}, best_name)

def shuffle_train_keys(train_keys):
    """shuffle train keys by protein"""
    sample_dict = defaultdict(list)
    for i in train_keys:
        key = i.split('/')[-1].split('_')[0]
        sample_dict[key].append(i)
    keys = list(sample_dict.keys())
    np.random.shuffle(keys)
    new_keys = []
    batch_sizes = []

    for i,key in enumerate(keys):
        temp = sample_dict[key]
        np.random.shuffle(temp)
        new_keys += temp
        batch_sizes.append(len(temp))
    return new_keys,batch_sizes
