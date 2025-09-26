import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import optimizer
from tqdm import tqdm
import json
import os
import gc
import random

import pickle

from torch.utils.data import DataLoader,TensorDataset

import argparse
from src.utils import setup_seed, multi_acc
from src.pixel_classifier import  load_ensemble, compute_iou, predict_labels, save_predictions,calculate_metric_per_class_plot_cm, pixel_classifier
from src.datasets import ImageLabelDataset, FeatureDataset, make_transform,BerlinFusionDataset,AugsburgFusionDataset
from src.feature_extractors import create_feature_extractor, collect_features
from src.plot_tsne import compute_all_pairwise_similarities_df,plot_all_pairs_boxplots
from src.cka import compute_pairwise_cka_df
from src.data_util import get_class_names

from guided_diffusion.guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from guided_diffusion.guided_diffusion.dist_util import dev
from guided_diffusion.guided_diffusion.logger import  configure, logkv, dumpkvs,log
import pdb

import faiss

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd


from matplotlib.backends.backend_pdf import PdfPages

from torch.utils.data import random_split



def to_numpy(tensor):
    """Convert PyTorch tensor to NumPy array if needed."""
    return tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor

def to_tensor(array, device="cpu"):
    """Convert NumPy array back to PyTorch tensor if needed."""
    return torch.tensor(array, dtype=torch.float32, device=device) if isinstance(array, np.ndarray) else array




def prepare_data(args,mode = "training"):
    feature_extractor = create_feature_extractor(**args)

    
    print(f"Preparing the {mode} set for {args['category']}...")

    if args["category"].lower() == "berlin":
        dataset = BerlinFusionDataset(
            data_dir=args[f"{mode}_path"],
            resolution=args['image_size'],
            num_images=args[f"{mode}_number"],
            transform=make_transform(
                args['model_type'],
                args['image_size']
            )
        )
    elif args["category"].lower() == "augsburg":
        dataset = AugsburgFusionDataset(
            data_dir=args[f"{mode}_path"],
            resolution=args['image_size'],
            num_images=args[f"{mode}_number"],
            transform=make_transform(
                args['model_type'],
                args['image_size']
            )
        )
    else:
        raise ValueError(f"Unsupported category: {args['category']}")

    
    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=dev())
    else:
        noise = None 
    
    pixel_num  = torch.tensor(0)    


    # Map modality index -> (name, class_id)
    modality_map = {
            0: ("rgb", 1),
            1: ("pca", 2),
            2: ("sar", 3),
            3: ("dsm", 4)
    }
        
    for row, sample in enumerate(tqdm(dataset)):
        
        label = sample["label"]
        X_spatial_list = []

  
        for mod_idx, (mod_name, class_id) in modality_map.items():
            if mod_idx in args["use_modalities"] and mod_name in sample:
                img = sample[mod_name][None].to(dev())  # Add batch dimension
                y = torch.full((1,), class_id).to(dev())  # Assign class ID for conditioning
                features = feature_extractor(img, noise=noise, y=y)
                X_spatial_list.append(collect_features(args, features).cpu())
        X_spatial = torch.cat(X_spatial_list, dim=1)

        B, C, H, W = X_spatial.shape
        X_spatial = X_spatial.permute(0, 2, 3, 1).reshape(B*H*W, C)

        y = label       

        y = y.flatten()
        
        X_spatial = X_spatial[y != args['ignore_label']]
        y = y[y != args['ignore_label']]
        
        #Concantenate the X and y
        if row == 0:
          concan_X_spatial = X_spatial
          concan_y = y
          
        else:
          concan_X_spatial = torch.cat((concan_X_spatial, X_spatial), dim  = 0)
          concan_y = torch.cat((concan_y, y),dim  =0 )

    return concan_X_spatial, concan_y



def evaluation(args, models):
    feature_extractor = create_feature_extractor(**args)

    if args["category"].lower() == "berlin":
        dataset = BerlinFusionDataset(
            data_dir=args["testing_path"],
            resolution=args['image_size'],
            num_images=args["testing_number"],
            transform=make_transform(
                args['model_type'],
                args['image_size']
            )
        )
    elif args["category"].lower() == "augsburg":
        dataset = AugsburgFusionDataset(
            data_dir=args["testing_path"],
            resolution=args['image_size'],
            num_images=args["testing_number"],
            transform=make_transform(
                args['model_type'],
                args['image_size']
            )
        )
    else:
        raise ValueError(f"Unsupported category: {args['category']}")
        
    #pdb.set_trace()
    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=dev())
    else:
        noise = None 

    preds, gts, uncertainty_scores = [], [], []
    image_paths = []


    modality_map = {
    0: ("rgb", 1),
    1: ("pca", 2),
    2: ("sar", 3),
    3: ("dsm", 4) } # Only used if Augsburg has DSM

    for sample in tqdm(dataset):
        
        label = sample['label']  

        x_spatial_list = []
        
        for mod_idx, (key, class_id) in modality_map.items():
            if mod_idx in args["use_modalities"] and key in sample:
                img = sample[key][None].to(dev())  # Add batch dim
                y_mod = torch.full((1,), class_id).to(dev())
                feat_mod = feature_extractor(img, noise=noise, y=y_mod)
                x_spatial_list.append(collect_features(args, feat_mod))

        x_spatial = torch.cat(x_spatial_list, dim=1)
    
        # Reshape to [H*W, C]
        #x_spatial = x_spatial.view(args['dim'][-1], -1).permute(1, 0)
        B, C, H, W = x_spatial.shape
        x_spatial = x_spatial.permute(0, 2, 3, 1).reshape(B * H * W, C)


        index = sample['index']
        image_paths.append(f"hyperspectral{str(index)}.png")

     
        pred = predict_labels(
            models, x_spatial, size=args['dim'][:-1]
        )

        #for softvoting ,pred is shape  (H, W, num_classes) 
        gts.append(label.numpy()-1)
        preds.append(pred.numpy())
       # uncertainty_scores.append(uncertainty_score.item())
    
    test_label_filted,inference_map_filted= save_predictions(args, image_paths, preds)
    test_label = np.load(args['test_label_path'])
    calculate_metric_per_class_plot_cm(args,test_label_filted,inference_map_filted)


def mixup_data(x_spatial, y, alpha=0.2, device='cpu'):


    if alpha > 0:
        lam_spatial = np.random.beta(alpha, alpha)
    else:
        lam_spatial = 1
    lam = lam_spatial  # No spectral component, so use only lam_spatial

    batch_size = x_spatial.size(0)
    index = torch.randperm(batch_size).to(device)


    mixed_x_spatial = lam_spatial * x_spatial + (1 - lam_spatial) * x_spatial[index, :]

    y_one_hot = torch.nn.functional.one_hot(y, num_classes=8).float()
    
    # Apply MixUp to labels
    mixed_y = lam * y_one_hot + (1 - lam) * y_one_hot[index, :]

    return mixed_x_spatial, mixed_y

def mixup_criterion(pred, mixed_y):

    log_softmax = torch.nn.functional.log_softmax(pred, dim=1)
    loss = -torch.sum(mixed_y * log_softmax, dim=1)
    return torch.mean(loss)


## log learnable modality weight
def log_modality_weights(model, log_fn=None):
    if hasattr(model, 'alpha'):
        with torch.no_grad():
            weights = F.softmax(model.alpha, dim=0).detach().cpu().numpy()
        names = ["RGB", "PCA", "SAR", "DSM"][:len(weights)]
        message = "Learned modality weights:\n" + "\n".join(
            f"  {n:10s}: {w:.4f}" for n, w in zip(names, weights)
        )
        print(message)
        if log_fn:
            log_fn(message)



def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Runs one epoch of training.
    """
    model.train()
    total_loss = 0
    total_acc = 0
    use_mixup = False
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_batch = y_batch.type(torch.long)
        optimizer.zero_grad()
        
        if use_mixup:
            mixed_X, mixed_y = mixup_data(X_batch, y_batch, alpha=0.1, device=device)
            y_pred = model(X_batch)
            loss = mixup_criterion(y_pred, mixed_y)
        else:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch - 1)  # labels are 1-indexed
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += multi_acc(y_pred, y_batch - 1)

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)

    return avg_loss, avg_acc

def val_one_epoch(model, val_loader, criterion, device):
    """
    Runs one epoch of validation.
    """
    model.eval()
    total_loss = 0
    total_acc = 0
    critetion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_batch = y_batch.type(torch.long)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch - 1)

            total_loss += loss.item()
            total_acc += multi_acc(y_pred, y_batch - 1)

    avg_loss = total_loss / len(val_loader)
    avg_acc = total_acc / len(val_loader)

    return avg_loss, avg_acc






from src.alignment import *
from src.losses import entropy_loss 
from src.plot_tsne import plot_tsne,plot_tsne_overlay,plot_tsne_by_modality,compute_modality_cosine_similarity,plot_cosine_similarity_boxplot,compute_and_display_class_stats
def train(args):
    
    configure(dir=args['exp_dir'], format_strs=["log"])

    log(
        f"Preparing the train set for {args['category']} with learning rate {args['learning_rate']}, "
        f"{args['blocks']} layers, {args['steps']} timesteps, batch size {args['batch_size']}, "
        f"training path: {args['training_path']}, testing path: {args['testing_path']}, "
        f"input activations: {args['input_activations']}, model path: {args['model_path']}, "
        f"max epochs: {args['max_epoch']}, mix_up_alpha: {args['mix_up_alpha']}, "
        f"use_mixup: {args['use_mixup']}, voting method: {args['voting_type']}."
        f"use_modalities: {args['use_modalities']}."

    )


    
    device = dev()  # Get device only once
    
    
    ############comment out this part when saved ###########
    train_features, train_labels = prepare_data(args, mode="training")

    test_features, test_labels = prepare_data(args, mode="testing")
    

    train_data = TensorDataset(train_features, train_labels)
    test_data = TensorDataset(test_features, test_labels)

    val_frac = 0.3  # Fraction of test_data to use for validation
    subset_size = int(val_frac * len(test_data))  # Compute validation dataset size

    split_gen = torch.Generator(device="cpu").manual_seed(args['seed'])


    
    _, test_subset = random_split(
            test_data,
            [len(test_data) - subset_size, subset_size],
            generator=split_gen
        )

    log(f"Using {subset_size} samples ({val_frac*100:.1f}%) from test_data as validation dataset.")



    ###############################################

    #train_data = torch.load('Berlin_train_sar_pca_rgb.pt')
    
    #test_subset = torch.load('Berlin_val_sar_pca_rgb.pt')

    #train_features, train_labels = train_data.tensors
 
    # compute per clas per pair modality similarity
    if args["use_modalities"] == [0,1,2]:
        df_train = compute_all_pairwise_similarities_df(
        train_features, train_labels,
        use_modalities=args["use_modalities"])
        # save train CSV
        csv_train_dir = os.path.join(args["exp_dir"], "pairwise_cosine_similarities_train.csv")
        df_train.to_csv(csv_train_dir, index=False)
        print(f"Saved: {csv_train_dir}")

        class_names = get_class_names(args["category"])
        #class_names = ["Forest","Residential","Industrial","Low Plants","Soil","Allotment","Commercial","Water"]
        plot_all_pairs_boxplots(df_train, class_names=class_names, save_path=args["exp_dir"])

        df_cka_train = compute_pairwise_cka_df(train_features,use_modalities=args["use_modalities"])
        log("CKA (train):\n{}".format(df_cka_train.round(4).to_string(index=False)))
        
    

    print(f" ********* max_label {args['number_class']} *** ignore_label {args['ignore_label']} ***********")

    # Create data loaders
    g_train = torch.Generator(device="cpu").manual_seed(args['seed'])
    g_val   = torch.Generator(device="cpu").manual_seed(args['seed'])


    train_loader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True,
                          generator=g_train, num_workers=0, drop_last= False)

    val_loader   = DataLoader(test_subset, batch_size=args['batch_size'], shuffle= True,
                          generator=g_val,   num_workers=0, drop_last= True)

    #train_loader = DataLoader(dataset=train_data, batch_size=args['batch_size'], shuffle=True, drop_last= True)
    #val_loader = DataLoader(dataset=test_subset, batch_size=args['batch_size'], shuffle=True, drop_last=False)

    print(f" *********************** Training dataloader length: {len(train_loader)} ***********************")
    print(f" *********************** Validation dataloader length: {len(val_loader)} ***********************")



    for MODEL_NUMBER in range(args['start_model_num'], args['model_num']):
        gc.collect()
        
        model = pixel_classifier(num_classes=args['number_class'], fused_dim=args['dim'][-1],
                                 use_modalities = args["use_modalities"],category=args["category"])
        #model = pixel_classifier(num_classes=args['number_class'], fused_dim=args['dim'][-1],use_modalities = [0,1,2])
        model.to(device)


        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'],weight_decay=5e-4)
        iteration = 0
        break_count = 0
        best_loss = 10000000
        best_acc = 0
        stop_sign = 0
        alpha = args['mix_up_alpha']

        for epoch in range(100):
            model.train()
            total_loss = 0
            for X_spatial_batch, y_batch in train_loader:
                #pdb.set_trace()
                X_spatial_batch, y_batch = X_spatial_batch.to(dev()), y_batch.to(dev())
                y_batch = y_batch.type(torch.long)

                if(args["use_mixup"]):
                    X_spatial_batch, y_batch_mixed = mixup_data(X_spatial_batch, y_batch-1, alpha=alpha, device=dev())
                
                
                # Forward pass through the combined model
                y_pred = model(X_spatial_batch)

                if(args["use_mixup"]):
                    loss = mixup_criterion(y_pred, y_batch_mixed)
                else:
                    loss = criterion(y_pred, y_batch-1)

                #pdb.set_trace()
                optimizer.zero_grad()                      

                loss.backward()
                optimizer.step()
  
                #acc = multi_acc(y_pred, y_batch-1)
                iteration += 1
                if iteration % 30 == 0:
             
                    train_loss, train_acc = val_one_epoch(model, train_loader, criterion, device)
                    model.train()
                    print(f"Epoch: {epoch}, Iteration: {iteration}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
                    log(f"Epoch: {epoch}, Iteration: {iteration}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
                
                    if epoch > args['max_epoch']:
                        val_loss, val_acc = val_one_epoch(model, val_loader, criterion, device)
                        model.train()
                        print(f"Epoch: {epoch}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
                        log(f"Epoch: {epoch}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
                        #log_modality_weights(model, log)

                        if val_acc >  best_acc:
                            best_acc = val_acc
                            best_loss = val_loss
                            break_count = 0
                            model_path = os.path.join(args['exp_dir'], f'model_{MODEL_NUMBER}.pth')
                            torch.save({'model_state_dict': model.state_dict()}, model_path)
                            print(f"Best model saved as model_{MODEL_NUMBER}.pth with validation loss: {val_loss:.4f}")
                            log(f"Best model saved as model_{MODEL_NUMBER}.pth with validation loss: {val_loss:.4f}")
                        else:
                            break_count += 1
                        if break_count > 10:
                            stop_sign = 1
                            message = f"*************** Break, Total iters: {iteration}, at Epoch: {epoch} ***************"
                            print(message) 
                            log(message)    

                            break

            if stop_sign == 1:
                break
        """
        model_path = os.path.join(args['exp_dir'], 
                                  'model_' + str(MODEL_NUMBER) + '.pth')
        MODEL_NUMBER += 1
        print('save to:',model_path)
        torch.save({'model_state_dict': model.state_dict()},
                   model_path)
        """

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())

    parser.add_argument('--exp', type=str)
    #parser.add_argument('--seed', type=int,  default=None)

    args = parser.parse_args()

    # Load the experiment config
    opts = json.load(open(args.exp, 'r'))
    opts.update(vars(args))

    # set seed
    if opts["seed"] is None:   # fallback if missing
        opts["seed"] = 0

    setup_seed(opts["seed"]) 
    
    opts['image_size'] = opts['dim'][0]
    

    '''
    if len(opts['steps']) > 0:
        suffix = '_'.join([str(step) for step in opts['steps']])
        suffix += '_' + '_'.join([str(step) for step in opts['blocks']])
        opts['exp_dir'] = os.path.join(opts['exp_dir'], suffix)
    '''
    # Prepare the experiment folder 
    if len(opts['steps']) > 0:
        suffix = '_'.join([str(step) for step in opts['steps']])
        suffix += '_' + '_'.join([str(step) for step in opts['blocks']])
    
        # include seed in folder name

        if 'seed' in opts:
            suffix += f"_seed{opts['seed']}"
        else:
            suffix += "_seed0"   # fallback if missing
    
        opts['exp_dir'] = os.path.join(opts['exp_dir'], suffix)


    path = opts['exp_dir']
    os.makedirs(path, exist_ok=True)
    print('Experiment folder: %s' % (path))
    os.system('cp %s %s' % (args.exp, opts['exp_dir']))

    # Check whether all models in ensemble are trained 
    pretrained = [os.path.exists(os.path.join(opts['exp_dir'], f'model_{i}.pth')) 
                  for i in range(opts['model_num'])]
              
    if not all(pretrained):
        # train all remaining models
        opts['start_model_num'] = sum(pretrained)
        train(opts)
    
    print('Loading pretrained models...')
    models = load_ensemble(opts, device='cuda')
    evaluation(opts, models)
