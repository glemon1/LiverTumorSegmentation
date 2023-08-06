# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 10:47:45 2022

@author: glemon
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import shutil

from numpy import asarray
import nibabel as nib
import cv2
from tqdm.notebook import tqdm
# from ipywidgets import *
from PIL import Image

from fastai.basics import *
from fastai.vision.all import *
from fastai.data.transforms import *



#%% Load saved model
im_size = 128

# the labels used for the classes
# When predicting the model predicts it in terms of indices (ie 0 --> background, 1 --> liver ...)
codes = np.array(["background","liver","tumor"])

# the default pathb
path = './'

def get_x(fname:Path): 
    return fname

def label_func(x): 
    return path/'train_masks'/f'{x.stem}_mask.png'

def foreground_acc(inp, targ, bkg_idx=0, axis=1):  # exclude a background from metric
    "Computes non-background accuracy for multiclass segmentation"
    targ = targ.squeeze(1)#removes dimension 1 from targ (removes [])
    mask = targ != bkg_idx
    return (inp.argmax(dim=axis)[mask]==targ[mask]).float().mean() 

def cust_foreground_acc(inp, targ):  # # include a background into the metric
    return foreground_acc(inp=inp, targ=targ, bkg_idx=3, axis=1) # 3 is a dummy value to include the background which is 0



#%%extras from preproccesing (redefined here)

dicom_windows = types.SimpleNamespace(
    brain=(80,40),
    subdural=(254,100),
    stroke=(8,32),
    brain_bone=(2800,600),
    brain_soft=(375,40),
    lungs=(1500,-600),
    mediastinum=(350,50),
    abdomen_soft=(400,50),
    liver=(150,30),
    spine_soft=(250,50),
    spine_bone=(1800,400),
    custom = (200,60)
)

#% Preprocessing functions
# Source https://docs.fast.ai/medical.imaging

class TensorCTScan(TensorImageBW): _show_args = {'cmap':'bone'}

@patch
def freqhist_bins(self:Tensor, n_bins=100):
    "A function to split the range of pixel values into groups, such that each group has around the same number of pixels"
    imsd = self.view(-1).sort()[0]
    t = torch.cat([tensor([0.001]),
                   torch.arange(n_bins).float()/n_bins+(1/2/n_bins),
                   tensor([0.999])])
    t = (len(imsd)*t).long()
    return imsd[t].unique()
    
@patch
def hist_scaled(self:Tensor, brks=None):
    "Scales a tensor using `freqhist_bins` to values between 0 and 1"
    if self.device.type=='cuda': return self.hist_scaled_pt(brks)
    if brks is None: brks = self.freqhist_bins()
    ys = np.linspace(0., 1., len(brks))
    x = self.numpy().flatten()
    x = np.interp(x, brks.numpy(), ys)
    return tensor(x).reshape(self.shape).clamp(0.,1.)
    
    
@patch
def to_nchan(x:Tensor, wins, bins=None):
    "Takes a tensor or a dicom as the input and returns multiple one channel images. Setting bins to 0 only returns the windowed image."
    res = [x.windowed(*win) for win in wins]
    if not isinstance(bins,int) or bins!=0: res.append(x.hist_scaled(bins).clamp(0,1))
    dim = [0,1][x.dim()==3]
    return TensorCTScan(torch.stack(res, dim=dim))

@patch
def save_jpg(x:(Tensor), path, wins, bins=None, quality=90):
    fn = Path(path).with_suffix('.jpg')
    x = (x.to_nchan(wins, bins)*255).byte()
    im = Image.fromarray(x.permute(1,2,0).numpy(), mode=['RGB','CMYK'][x.shape[0]==4])
    im.save(fn, quality=quality)
    
    
@patch
def windowed(self:Tensor, w, l):
    px = self.clone()
    px_min = l - w//2
    px_max = l + w//2
    px[px<px_min] = px_min
    px[px>px_max] = px_max
    return (px-px_min) / (px_max-px_min)


#plotting
def plot_sample_pred(array_list, color_map='nipy_spectral'):
    '''
    Plots and a slice with all available annotations
    '''
    fig = plt.figure(figsize=(18,15))

    plt.subplot(1,6,1)
    plt.imshow(array_list[0], cmap='bone')
    plt.title('Original Image')
    
    plt.subplot(1,6,2)
    #plt.imshow(tensor(array_list[1].astype(np.float32)).windowed(*dicom_windows.liver), cmap='bone')
    plt.imshow(array_list[1], cmap='gray', vmin=0, vmax=255)
    plt.title('Mask')
    
    plt.subplot(1, 6, 3)
    plt.imshow(array_list[2], cmap='gray', vmin=0, vmax=255) #cmap=color_map)
    plt.title('Pred Mask')
    
    plt.subplot(1,6,4)
    plt.imshow(array_list[0], cmap='bone')
    plt.imshow(array_list[1], alpha=0.5, cmap=color_map)
    plt.title('Liver & Mask')
    
    plt.subplot(1,6,5)
    plt.imshow(array_list[0], cmap='bone')
    plt.imshow(array_list[2], alpha=0.5, cmap='gray')#color_map
    plt.title('Liver & pred Mask')

    plt.subplot(1, 6, 6)
    #plt.imshow(array_list[1], cmap=color_map)
    plt.imshow(array_list[1], cmap='gray', vmin=0, vmax=255) 
    plt.imshow(array_list[2], alpha=0.3, cmap=color_map)
    plt.title('Mask & Pred Mask')

    plt.show()
    
#end extras

#Getting predictions done on multiple images.
#../input/trained-model
def nii_tfm_selctive(fn,wins,curr_slice): 
    slices = []
    test_nii  = read_nii(fn)
    data = tensor(test_nii[...,curr_slice].astype(np.float32))
    data = (data.to_nchan(wins)*255).byte()
    slices.append(TensorImage(data))
    return slices

def check(img):
    cnt,h = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(cnt) > 0:
        return 1
    else:
        return 0

#based on these add functio to calculate TP, TN, FP, FN etc for each value o, 1, 2 in mask arrays. Separate and compare each one alone by eg mask = targ != bkg_idx, mask = targ==2 for tumor, 1 for liver, >0 for 1 and 2
#mask = targ ==2; pred = pred ==2, count pred == mask to see how many are correct TP , divide by count mask,  
def metrics_segm(target, pred):
    metrics = dict();
    #target1s = np.count_nonzero(target == 1) # count one s in target for liver
    #target2s = np.count_nonzero(target == 2)  # count two s in target for tumor
    #pred1s = np.count_nonzero(pred == 1)
    #pred2s = np.count_nonzero(pred == 2)
    #calculate one for liver, one for tumor 
    #if target1s > 0 or  pred1s >0:
    targ_liver =  target == 1
    pred_liver =  pred == 1
    intersection_liver = np.logical_and(targ_liver, pred_liver) # TP
    union_liver = np.logical_or(targ_liver, pred_liver)
    sum_union_liver = np.sum(union_liver) # 𝑇𝑃 + 𝐹𝑃 + 𝐹𝑁 : all positive values, either in pred or in mask or both
    if sum_union_liver>0:
        iou_score_liver = np.sum(intersection_liver) / sum_union_liver
    else:
        iou_score_liver=-1
        
    #if target2s > 0 or  pred2s >0:
    targ_tum =  target == 2
    pred_tum =  pred == 2
    
    intersection_tum = np.logical_and(targ_tum, pred_tum)
    union_tum = np.logical_or(targ_tum, pred_tum)
    sum_union_tum = np.sum(union_tum)
    if sum_union_tum>0:
        iou_score_tum = np.sum(intersection_tum) / sum_union_tum 
    else:
        iou_score_tum=-1   # no metric for tumor 
    #add TP TN, FP, FN, F1, preccision, recall for tumor and liver
    
    metrics["iou_score_liver"]= iou_score_liver
    metrics["iou_score_tum"]= iou_score_tum
    return metrics
    
def get_metrics(y_true, y_pred, loss=float('nan')):
    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    accuracy = (TP + TN) / (len(y_true)*len(y_true[0])) # for two dimension: divide by /rows*cols to get total items, for one dim len(y_true) is enough
    assert not np.isnan(TN)
    assert not np.isnan(FP)
    assert not np.isnan(FN)
    assert not np.isnan(TP)

    if TP > 0:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * ((precision * recall) / (precision + recall))
        iou = TP / (TP + FP + FN)
    else:
        precision = recall = f1_score = iou = float('NaN')
    return {
        "accuracy": accuracy,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "iou": iou,
        "true_mean": np.mean(y_true),
        "pred_mean": np.mean(y_pred),
        "loss": loss
    } 

 
def get_metrics_liver(y_true, y_pred): #https://www.sciencedirect.com/science/article/pii/S0010482521002912 metrics from
    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    TN = np.sum(np.logical_and(y_pred == 0, np.logical_or(y_true == 0, y_true == 2 )))
    FP = np.sum(np.logical_and(y_pred == 1, np.logical_or(y_true == 0, y_true == 2 )))
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    accuracy = (TP + TN) / (TP + TN+FP+FN)#(len(y_true)*len(y_true[0])) # /rows*cols to get total items
    assert not np.isnan(TN)
    assert not np.isnan(FP)
    assert not np.isnan(FN)
    assert not np.isnan(TP)

    if (TP + FP) > 0:
        precision = TP / (TP + FP) #or PPV
        Sens = precision # TP / (TP + FP) # Sensitivity sensitivity is the same ass preccision
        FPR =  FP / (FP + TP)#mine FP rate compared to total Positive values, good for missdiagnosis when not exists liver or tumor, use TP in divider because TN are many
    else:
        precision = Sens = FPR = float('NaN')

    if (TP + FN) > 0:
        recall = TP / (TP + FN) 
        FNR =  FN / (FN + TP)# FN rate compared to total Positive values, good to detect if miss the liver or tumor, use TP in divider because TN are many
    else:
        recall = FNR = float('NaN')

    if (precision + recall) > 0:
        f1_score = 2 * ((precision * recall) / (precision + recall))
    else:
        f1_score = float('NaN')  

    if (TP + FP + FN) > 0: 
        iou = TP / (TP + FP + FN)
        DSC = 2 * TP / (2*TP + FP + FN) # Dice Similarity Coefficient    
    else:
        iou = DSC = float('NaN')
        
    if  (TN + FP) > 0:   
        Spec = TN / (TN + FP) # Specificity
    else:
        Spec = float('NaN')
        
    return {
        "accuracy": accuracy,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "iou": iou,
        "DSC": DSC,
        "FNR": FNR,
        "FPR": FPR,
        "Sens": Sens,
        "Spec": Spec
    } 

def get_metrics_tum(y_true, y_pred):
    TP = np.sum(np.logical_and(y_pred == 2, y_true == 2))
    TN = np.sum(np.logical_and(y_pred == 0, np.logical_or(y_true == 0, y_true == 1)))
    FP = np.sum(np.logical_and(y_pred == 2, np.logical_or(y_true == 0, y_true == 1)))
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 2))
    accuracy = (TP + TN) / (TP + TN+FP+FN)#(len(y_true)*len(y_true[0])) # /rows*cols to get total items
    assert not np.isnan(TN)
    assert not np.isnan(FP)
    assert not np.isnan(FN)
    assert not np.isnan(TP)

    if (TP + FP) > 0:
        precision = TP / (TP + FP)
        Sens = precision # TP / (TP + FP) # Sensitivity sensitivity is the same ass preccision
        FPR =  FP / (FP + TP)#mine FP rate compared to total Positive values, good for missdiagnosis when not exists liver or tumor, use TP in divider because TN are many
    else:
        precision = Sens = FPR = float('NaN')

    if (TP + FN) > 0:
        recall = TP / (TP + FN)
        FNR =  FN / (FN + TP)# FN rate compared to total Positive values, good to detect if miss the liver or tumor, use TP in divider because TN are many
    else:
        recall = FNR = float('NaN')

    if (precision + recall) > 0:
        f1_score = 2 * ((precision * recall) / (precision + recall))
    else:
        f1_score = float('NaN')  

    if (TP + FP + FN) > 0: 
        iou = TP / (TP + FP + FN)
        DSC = 2 * TP / (2*TP + FP + FN) # Dice Similarity Coefficient    
    else:
        iou = DSC = float('NaN')
        
    if  (TN + FP) > 0:   
        Spec = TN / (TN + FP) # Specificity
    else:
        Spec = float('NaN')
        
    return {
        "accuracy": accuracy,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "iou": iou,
        "DSC": DSC,
        "FNR": FNR,
        "FPR": FPR,
        "Sens": Sens,
        "Spec": Spec
    } 



#%%loop start loop  for train_masks
file_iterator = os.scandir('./train_images') 
for i in file_iterator: #create and save train masks, liver and tumor and save train scans containing liver, rest ignore
    liverImg = cv2.imread("./train_images/"+i.name, cv2.IMREAD_UNCHANGED) #-1 unchanged
    liverFile = i.name

    if liverImg is None: 
        print('train liverImg empty or read error' + liverFile)

    maskFile = liverFile.split(".")[0] + "_mask.png"
    maskImg = cv2.imread("./train_masks/" + maskFile, cv2.IMREAD_GRAYSCALE) #0 grayscale
    if maskImg is None: 
        print('train MaskImg empty or read error' + maskFile)

    npmaskImg=np.array(maskImg)
    npLiverImg=np.array(liverImg)

    print("train mask: "+ maskFile)
    if np.any(npmaskImg):#if non zero mask create and save images for test and train images
        print('non zero mask')
        npLiverMask =  (npmaskImg > 0).astype(int) #put 1 where 1 or 2 exist
        npTumMask =  (npmaskImg == 2).astype(int) # put 1 where 2 exists
        
        liver_mask_img = Image.fromarray(npLiverMask.astype(np.uint8)) #save mask with liver (and tumor as liver)
        liver_mask_img.save(f"train_masks_liver/{maskFile}")
        #if np.any(npTumMask):#I guess for train we must have tum mask with zeroes 
        tum_mask_img = Image.fromarray(npTumMask.astype(np.uint8)) #save mask with tumor only
        tum_mask_img.save(f"train_masks_tumor/{maskFile}")

        # In train save only-- liverImg.save(f"train_images_liver/{liverFile}")#train images containing liver, non liver images ignored
        npLiverMask3chanel = np.zeros_like(liverImg)   # create a 3 channel mask
        npLiverMask255 = np.where(npLiverMask ==1, 255,0) # mask must have value 255 to work
        npLiverMask3chanel[:,:,0] = npLiverMask255
        npLiverMask3chanel[:,:,1] = npLiverMask255
        npLiverMask3chanel[:,:,2] = npLiverMask255
        npliverImgMasked = cv2.bitwise_and(liverImg, npLiverMask3chanel)
        
        liverImgMasked = Image.fromarray(npliverImgMasked.astype(np.uint8)) #save masked liver image
        liverImgMasked.save(f"train_images_liver_masked/{liverFile}") # for train model finding tumors from masked liver images
        
        #save train image only if contains liver (copy from train_images to train_images_liver--> only images with liver for better train)
        src_path = f"train_images/{liverFile}"
        #src_path = r"E:\demos\files\report\profit.txt"
        dst_path = f"train_images_liver/{liverFile}"
        shutil.copy(src_path, dst_path)

        npLiverMaskLight = npLiverMask*120 # multiply it to make it brighter and visible to plot
        npTumMaskLight = npTumMask*120
        npliverImgMasked=np.array(liverImgMasked)

        print(np.unique(npmaskImg, return_counts=True))

        # plt.imshow(npliverImgMasked)
        # plt.show()
        #plot_sample_pred([npliverImgMasked, npLiverMaskLight, npTumMaskLight]) #plot ct, mask and pred_mask, one on another
    else:
        print('no liver in test image ' + liverFile)
        
    # train loop end
#%%loop for test_masks   
file_iterator = os.scandir('./test_images') 
for i in file_iterator: #create and save test masks, liver and tumor and save test scans containing liver, rest ignore
    liverImg = cv2.imread("./test_images/"+i.name, cv2.IMREAD_UNCHANGED) #-1 unchanged
    liverFile = i.name
    if liverImg is None: 
        print('liverImg empty or read error')

    maskFile = liverFile.split(".")[0] + "_mask.png"
    maskImg = cv2.imread("./test_masks/" + maskFile, cv2.IMREAD_GRAYSCALE) #0 grayscale
    if maskImg is None: 
        print('MaskImg empty or read error')

    npmaskImg=np.array(maskImg)
    npLiverImg=np.array(liverImg)

    print("test mask file: " + maskFile)
    if np.any(npmaskImg):#if non zero mask create and save images for test and train images
        print('non zero mask')
        npLiverMask =  (npmaskImg > 0).astype(int)
        npTumMask =  (npmaskImg == 2).astype(int)
        
        liver_mask_img = Image.fromarray(npLiverMask.astype(np.uint8)) #save mask with liver (and tumor as liver)
        liver_mask_img.save(f"test_masks_liver/{maskFile}")
        
        if np.any(npTumMask): # if tum exists, else no save tum mask, its all 0
            tum_mask_img = Image.fromarray(npTumMask.astype(np.uint8)) #save mask with tumor only
            tum_mask_img.save(f"test_masks_tumor/{maskFile}")

        # for test we use all liver images so no need to save. In train sav only-- liverImg.save(f"train_images_liver/{liverFile}")#train images containing liver, non liver images ignored
        npLiverMask3chanel = np.zeros_like(liverImg)   # create a 3 channel mask
        npLiverMask255 = np.where(npLiverMask ==1, 255,0) # mask must have value 255 to work
        npLiverMask3chanel[:,:,0] = npLiverMask255
        npLiverMask3chanel[:,:,1] = npLiverMask255
        npLiverMask3chanel[:,:,2] = npLiverMask255
        npliverImgMasked = cv2.bitwise_and(liverImg, npLiverMask3chanel)
        
        liverImgMasked = Image.fromarray(npliverImgMasked.astype(np.uint8)) #save masked liver image
        liverImgMasked.save(f"test_images_liver_masked/{liverFile}")

        npLiverMaskLight = npLiverMask*120 # multiply it to make it brighter and visible to plot
        npTumMaskLight = npTumMask*120
        npliverImgMasked=np.array(liverImgMasked)

    else:
        print('no liver in test image ' + liverFile)
    
        # for test, test with all test_images, even non liver, on testing  if masks liver n tumor not exist, they re zero
        # liverImg.save(f"test_images_liver/{liverFile}")#test images containing liver, non liver images ignored
       