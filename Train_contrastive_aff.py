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
os.environ['CUDA_LAUNCH_BLOCKING']='1'
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
RDLogger.DisableLog('rdApp.*')
s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
print (s)
os.chdir(os.path.abspath(os.path.dirname(__file__)))
def resample_data_max(pdb_groups, target_count=5, max_per_pdb=5):
    resampled_data = []
    for pdb_id, files in pdb_groups.items():
        if len(files) < target_count:
            # 增加样本数量，但不超过 max_per_pdb
            additional_files = random.choices(files, k=min(target_count - len(files), max_per_pdb - len(files)))
            files.extend(additional_files)
        elif len(files) > target_count:
            # 减少样本数量，但不超过 max_per_pdb
            files = random.sample(files, min(target_count, max_per_pdb))
        resampled_data.extend(files)
    return resampled_data
def tenkf():
    csv_file = 'pdbbind_NL_cleaned.csv'
    df = pd.read_csv(csv_file)
    pdb_aff_dict = dict(zip(df['pdb'], df['neglog_aff']))
    pdbs = os.listdir("/public/home/qiang/jkwang/RNA_dock_score/data/affinify_sdf/")
    pdbs = [pdb for pdb in pdbs if pdb.endswith('sdf')]
    # print(len(pdbs))
    pdb_groups = defaultdict(list)
    for pdb in pdbs:
        # print(pdb)
        pdb_id = pdb[:4]
        pdb_groups[pdb_id].append(pdb)
    
    # 将分组后的 PDB ID 列表化
    # print(pdb_groups.keys())
    pdb_id_list = list(pdb_groups.keys())
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    folds = list(kf.split(pdb_id_list))
    # print(pdb_id_list)
    target_count = max(len(files) for files in pdb_groups.values())
    print(f"Target count (max number of samples per PDB ID): {target_count}")
    splits = []
    # print(pdb_groups)
    for i in range(10):
        # random_fold_index = random.randint(0, 9)
        # print(folds[i])
        test_pdb_ids = [pdb_id_list[i] for i in folds[i][1]]
        train_pdb_ids = [pdb_id_list[i] for i in folds[i][0]]
        # print(test_pdb_ids)
        # 根据 PDB ID 获取对应的文件列表
        train_pdbs = [pdb for pdb_id in train_pdb_ids for pdb in pdb_groups[pdb_id]]
        
        train_pdbs_resampled = resample_data_max({pdb_id: pdb_groups[pdb_id] for pdb_id in train_pdb_ids})
        test_pdbs = [pdb for pdb_id in test_pdb_ids for pdb in pdb_groups[pdb_id]]
        
        test_pdbs = [pdb for pdb in test_pdbs if '_0' in pdb]

        # 随机打乱训练集
        random.shuffle(train_pdbs_resampled)

        # 取1/10的数据作为验证集
        num_validation_samples = len(train_pdbs_resampled) // 10
        valid_pdbs_resampled = train_pdbs_resampled[:num_validation_samples]

        # 剩余的作为新的训练集
        train_pdbs_resampled = train_pdbs_resampled[num_validation_samples:]
        # train_pdbs = [pdb for pdb in train_pdbs if '_0' in pdb]
        # print(list(set(test_pdbs)))
        # print(train_pdbs_resampled)
        splits.append({
            "train": train_pdbs_resampled,
            "test": list(set(test_pdbs)),
            "valid":list(set(valid_pdbs_resampled))
        })
        
    return splits
def run(local_rank,args):
    
    args.local_rank = local_rank
    args.lmdb_cache = "lmdbs/aff_pocket_draw"
    
    # torch.distributed.init_process_group(backend="nccl",init_method='env://',rank = args.local_rank,world_size = args.ngpu)  # multi gpus training，'nccl' mode
    torch.cuda.set_device(args.local_rank) 
    seed_torch(seed = args.seed + args.local_rank)
    # use attentiveFP feature or not
    if args.FP:
        args.N_atom_features = 39
    else:
        args.N_atom_features = 28
    num_epochs = args.epoch
    lr = args.lr
    save_dir = args.save_dir
    train_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    # make save dir if it doesn't exist
    if args.hot_start:
        if os.path.exists(args.save_model):
            best_name = args.save_model
            model_name = best_name.split('/')[-1]
            save_path = best_name.replace(model_name,'')
        else:
            raise ValueError('save_model is not a valid file check it again!')
    else:
        save_path = os.path.join(save_dir,'constrastive',train_time)
    # save_path = 'workdir/contrastive/2024-11-24-14-12-38'
    if not os.path.exists(save_path):
        os.system('mkdir -p ' + save_path)
    log_path = save_path+'/logs' 

    #read data. data is stored in format of dictionary. Each key has information about protein-ligand complex.

    # if args.train_val_mode == 'uniport_cluster':
    #     with open ("train_error.pkl", 'rb') as fp:
    #         train_error = list(pickle.load(fp))
    #     with open ("test_error.pkl", 'rb') as fp:
    #         test_error = list(pickle.load(fp))
    #     with open ("rdock_2013.pkl", 'rb') as fp:
    #         train2013_keys = list(pickle.load(fp))
    #     with open (args.train_keys, 'rb') as fp:
    #         train_keys = list(pickle.load(fp))
    #     with open (args.val_keys, 'rb') as fp:
    #         val_keys = list(pickle.load(fp))
    #     with open (args.test_keys, 'rb') as fp:
    #         test_keys = list(pickle.load(fp))
    #     # with open ("testing_keys_10_2.pkl", 'rb') as fp:
    #     #     test_keys = list(pickle.load(fp))[:]
    # else:
    #     raise 'not implement this split mode,check the config file plz'
    import random
    
    with open("Contrastive_groups.pkl", "rb") as f:
        groups = pickle.load(f)
    train_groups = {}
    test_groups = {}
    splits = tenkf()
    # JSON文件路径
    splits_file = 'workdir/constrastive/2025-01-12-16-36-33/splits.json'
                  
    # 读取splits.json文件
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    with open(f'{save_path}/splits.json', 'w') as f:
        json.dump(splits, f, indent=4)
    # 根据 train_keys 和 test_keys 中的 PDB ID 将 groups 划分为测试组和训练组
    avg_mae = 0.0
    avg_rmse = 0.0
    avg_PCC = 0.0
    avg_SPCC = 0.0

    
    for i, split in enumerate(splits):
        # if i<6:
            # continue
        train_groups = {}
        test_groups = {}
        print(f"Split {i+1}:")

        train_keys = split["train"]
        test_keys = split["test"]
        val_keys = split["valid"]
        # 将 train_keys 中的 PDB ID 对应的 group 加入到 train_groups
        for key in train_keys:
            pdbid = key.split('_')[0]  # 提取四字符 PDB ID
            if pdbid in groups:
                if pdbid not in train_groups:
                    train_groups[pdbid] = groups[pdbid]

        # 将 test_keys 中的 PDB ID 对应的 group 加入到 test_groups
        for key in test_keys:
            pdbid = key.split('_')[0]  # 提取四字符 PDB ID
            if pdbid in groups:
                if pdbid not in test_groups:
                    test_groups[pdbid] = groups[pdbid]
        
        print(test_keys)
        # 打印训练组和测试组
            # print("Train Groups:", train_groups)
            # print("Test Groups:", test_groups)
        # return
        pos_keys = []
        neg_keys = []

        if local_rank == 0:
            print (f'Number of pos_keys data: {len(train_keys)}')
            print (f'Number of val data: {len(val_keys)}')
            print (f'Number of neg_keys data: {len(test_keys)}')

        model = EquiScore(args) if args.model == 'EquiScore' else GraphAttentionTransformer_dgl()
        print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        # args.device = args.local_rank
        args.device ='cuda' if torch.cuda.is_available() else 'cpu'
        if args.hot_start:
            model ,opt_dict,epoch_start= utils.initialize_model(model, args.device,args,args.save_model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

            optimizer.load_state_dict(opt_dict)

        else:
            # if args.model == 'EquiScore':
            model = utils.initialize_model(model, args.device,args)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            epoch_start = 0
            if i==0:
                write_log_head(args,log_path,model,pos_keys,neg_keys)
        # dataset processing
        train_dataset = ESDataset_contrastive(pos_keys,neg_keys, train_groups,args, args.data_path,args.debug)#keys,args, data_dir,debug

        val_dataset = ESDataset_aff(val_keys,args, args.data_path,args.debug)
        test_dataset = ESDataset_aff(test_keys,args, args.data_path,args.debug)
        # return
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.ngpu > 1 else None
        val_sampler = SequentialDistributedSampler(val_dataset,args.batch_size) if args.ngpu > 1 else None
        test_sampler = SequentialDistributedSampler(test_dataset,args.batch_size) if args.ngpu > 1 else None
        #    use sampler to balance the training data or not 
        if args.sampler:
            num_train_chembl = len([0 for k in train_keys if '_active' in k])
            num_train_decoy = len([0 for k in train_keys if '_active' not in k])
            train_weights = [1/num_train_chembl if '_active' in k else 1/num_train_decoy for k in train_keys]
            train_sampler = DTISampler(train_weights, len(train_weights), replacement=True)                     
            train_dataloader = DataLoaderX(train_dataset, args.batch_size, \
                shuffle=False, num_workers = args.num_workers, collate_fn=train_dataset.collate,prefetch_factor = 4,\
                sampler = train_sampler,pin_memory=True,drop_last = True) #dynamic sampler
        else:
            train_dataloader = DataLoaderX(train_dataset, args.batch_size, sampler = train_sampler,\
                shuffle=False, num_workers = args.num_workers, collate_fn=train_dataset.collate,pin_memory=True,prefetch_factor = 4)
        val_dataloader = DataLoaderX(val_dataset, args.batch_size, sampler=val_sampler,\
            shuffle=False, num_workers = args.num_workers, collate_fn=val_dataset.collate,pin_memory=True,prefetch_factor = 4)
        test_dataloader = DataLoaderX(test_dataset, args.batch_size, sampler=test_sampler,\
            shuffle=False, num_workers = args.num_workers, collate_fn=test_dataset.collate,pin_memory=True,prefetch_factor = 4) 

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr,pct_start=args.pct_start,\
             steps_per_epoch=len(train_dataloader), epochs=args.epoch,last_epoch = -1 if len(train_dataloader)*epoch_start == 0 else len(train_dataloader)*epoch_start )
        #loss function ,in this paper just use cross entropy loss but you can try focal loss too!
        if args.loss_fn == 'bce_loss':
            loss_fn = nn.BCELoss().to(args.device,non_blocking=True)# 
        elif args.loss_fn == 'focal_loss':
            loss_fn = FocalLoss().to(args.device,non_blocking=True)
        elif args.loss_fn == 'cross_entry':
            loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smothing).to(args.device,non_blocking=True)
        elif args.loss_fn == 'mse_loss':
            loss_fn = nn.MSELoss().to(args.device,non_blocking=True)
        elif args.loss_fn == 'poly_loss_ce':
            loss_fn = PolyLoss_CE(epsilon = args.eps).to(args.device,non_blocking=True)
        elif args.loss_fn == 'poly_loss_fl':
            loss_fn = PolyLoss_FL(epsilon=args.eps,gamma = 2.0).to(args.device,non_blocking=True)
        else:
            raise ValueError('not support this loss : %s'%args.loss_fn)

        best_loss = 1000000000
        best_f1 = -1
        counter = 0
        best_pcc = 0.0
        best_spcc = 0.0
        best_mae = 0.0
        best_rmse = 0.0

        
        # print(test_true)
        # print(test_pred)
        
        if args.onlytest:
            T = "2025-01-12-16-36-33"
            best_name = f"workdir/constrastive/{T}" + f'/save_best_model_{i}'+'.pt'
            if not os.path.exists(best_name):
                best_name =  f"workdir/constrastive/{T}" + f'/save_early_stop_model_{i}'+'.pt'
            
            checkpoint = torch.load(best_name)
            model.load_state_dict(checkpoint['model'])
            test_losses,test_true,test_pred,keys = testAndPrint_aff(model,test_dataloader,loss_fn,args,None)
            test_true = np.array(test_true).ravel()
            test_pred = np.array(test_pred).ravel()
            rmse = root_mean_squared_error(test_true, test_pred)
            mae = mean_absolute_error(test_true, test_pred)
            test_r_p = pearsonr(test_true, test_pred)[0]
            test_r_s = spearmanr(test_true, test_pred)[0]
            test_losses = torch.mean(torch.stack(test_losses), dim=0) 

            avg_mae += mae
            avg_PCC += test_r_p
            avg_rmse += rmse
            avg_SPCC += test_r_s

            test_file = f"workdir/constrastive/{T}" + '/test_results.txt'
            with open(test_file, 'a+') as out:
                out.write('{}\t{:.7f}\t{:.7f}\t{:.7f}\t{:.7f}\t'.format(
                    1111111, rmse, test_r_p, test_r_s, mae))
            continue
        for epoch in range(epoch_start,num_epochs):
            st = time.time()
            #collect losses of each iteration
            if args.ngpu > 1:
                train_sampler.set_epoch(epoch) 
            if args.use_gan:

                model,train_losses,loss_conts,loss_logits,optimizer,scheduler,d_loss = train_contrastive_gan(model,args,optimizer,loss_fn,train_dataloader,scheduler)
            else:
                model,train_losses,loss_conts,loss_logits,optimizer,scheduler = train_contrastive(model,args,optimizer,loss_fn,train_dataloader,scheduler)

            if args.ngpu > 1:
                dist.barrier() 
            val_losses,val_true,val_pred,keys= evaluator_aff(model,val_dataloader,loss_fn,args,val_sampler)

            if args.ngpu > 1:
                dist.barrier() 
            if local_rank == 0:
                # test_losses = 0.0
                # val_losses = 0.0
                
                train_losses = torch.mean(torch.tensor(train_losses,dtype=torch.float)).data.cpu().numpy()
                loss_cont = torch.mean(torch.tensor(loss_conts,dtype=torch.float)).data.cpu().numpy()
                loss_logit = torch.mean(torch.tensor(loss_logits,dtype=torch.float)).data.cpu().numpy()
                val_losses = torch.mean(torch.tensor(val_losses,dtype=torch.float)).data.cpu().numpy()
                if args.use_gan:
                    d_loss = torch.mean(torch.tensor(d_loss,dtype=torch.float)).data.cpu().numpy()
                if epoch%1==0:
                    test_losses,test_true,test_pred,keys = testAndPrint_aff(model,test_dataloader,loss_fn,args,None)
                    # print(test_true)
                    # print(test_pred)
                    
                    test_true = np.array(test_true).ravel()
                    test_pred = np.array(test_pred).ravel()
                    rmse = root_mean_squared_error(test_true, test_pred)
                    mae = mean_absolute_error(test_true, test_pred)
                    test_r_p = pearsonr(test_true, test_pred)[0]
                    test_r_s = spearmanr(test_true, test_pred)[0]
                    test_losses = torch.mean(torch.stack(test_losses), dim=0) 

                if args.loss_fn == 'mse_loss':
                    end = time.time()
                    with open(log_path,'a') as f:
                        if args.use_gan:
                            f.write(str(epoch)+ '\t'+str(train_losses)+ '\t'+str(val_losses)+ '\t'+str(test_losses) + '\t loss_cont:'+str(loss_cont) + '\t loss_logit:'+str(loss_logit) + '\t' + str(end-st)+'\t'+ f'test_Pearson R: {test_r_p:.7f}'+'\t'+f'test_Spearman R: {test_r_s:.7f}'+'\t'+f'test_RMSE: {rmse:.7f}'+'\t'+f'MAE: {mae:.7f}'+'\t'+f'd_loss: {d_loss:.7f}'+'\n')
                            f.close()
                        else:
                            f.write(str(epoch)+ '\t'+str(train_losses)+ '\t'+str(val_losses)+ '\t'+str(test_losses) + '\t loss_cont:'+str(loss_cont) + '\t loss_logit:'+str(loss_logit) + '\t' + str(end-st)+'\t'+ f'test_Pearson R: {test_r_p:.7f}'+'\t'+f'test_Spearman R: {test_r_s:.7f}'+'\t'+f'test_RMSE: {rmse:.7f}'+'\t'+f'MAE: {mae:.7f}'+'\t'+'\n')
                            f.close()
                else:
                    test_auroc,BEDROC,test_adjust_logauroc,test_auprc,test_balanced_acc,test_acc,test_precision,test_sensitity,test_specifity,test_f1 = get_metrics(val_true,val_pred)
                    end = time.time()
                    with open(log_path,'a') as f:
                        f.write(str(epoch)+ '\t'+str(train_losses)+ '\t'+str(val_losses)+ '\t'+str(test_losses) + '\t loss_cont:'+str(loss_cont) + '\t loss_cont:'+str(loss_logit) + '\t' + str(end-st)+ f'val_Pearson R: {val_r_p:.7f}' + '\t'+f'val_Spearman R: {val_r_s:.7f}'+'\t'+ f'test_Pearson R: {test_r_p:.7f}'+'\t'+f'test_Spearman R: {test_r_s:.7f}'+'\n')
                        f.close()
                counter +=1 
                if val_losses < best_loss:
                    best_loss = val_losses.item()
                    counter = 0
                    save_model(model,optimizer,args,epoch,save_path,i,mode = 'best')
                    best_mae = mae
                    best_rmse = rmse
                    best_pcc = test_r_p
                    best_spcc = test_r_s

                # if test_f1 > best_f1:
                #     best_f1 = test_f1
                #     counter = 0
                #     save_model(model,optimizer,args,epoch,save_path,mode = 'best_f1')
                if counter > args.patience:
                    save_model(model,optimizer,args,epoch,save_path,i,mode = 'early_stop')
                    print('model early stop !')
                    break
                if epoch == num_epochs-1:
                    save_model(model,optimizer,args,epoch,save_path,i,mode = 'end')
            if args.ngpu > 1:
                dist.barrier() 
        if args.ngpu > 1:
            dist.barrier() 
        avg_mae += best_mae
        avg_PCC += best_pcc
        avg_rmse += best_rmse
        avg_SPCC += best_spcc

    print(f"avg_mae:{avg_mae/10}, avg_PCC:{avg_PCC/10}, avg_rmse:{avg_rmse/10}, avg_SPCC:{avg_SPCC/10}")

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
    run(0,args)
    
    exit()
    
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


