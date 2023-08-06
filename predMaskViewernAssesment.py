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
    
    plt.savefig('foo.png')
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

def get_metrics(y_true, y_pred): #https://www.sciencedirect.com/science/article/pii/S0010482521002912 metrics from
    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))
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

# %% Model testing, one for test, final value:  (0, len(df_files))

#Unet outputs mask 128 image size (we resize also inputs to 128 to proccess them)
#originals are 512
#by scaling predictions to 512 I can compare
#but count pixels doesn t give metric for accuracy (stil 128 pred image haw less pixels)
# maybe if train without resize give better results but complexity, load;;; results look good event with resize
#the resize I do bellow to ipred mask image to become 512 from 128 probably decreases accuracy but .....


#%% Read and plot sample
imgDim = (512, 512) #dim mask Images should be compared, 512 original sizes, convert pred 128-->512

#Paths
path_test_im = './test_images' #3053
path_test_mask = './test_masks'
path_pred_mask = './pred_masks_livntum_trial_resnet50ep40L092T062'


#%%loop start
liverFile = "volume-98_slice_620.jpg"#volume-87_slice_100_mask.png

maskFile = liverFile.split(".")[0] + "_mask.png"
#if pred mask file not exists it means  it was full of zeros, it predicted background only:
predMaskFile = liverFile.split(".")[0] + "_mask.png"
liverImg = cv2.imread(path_test_im + "/" + liverFile)
#liverImg = cv2.imread("./test_images_liver_masked/" + liverFile)  # volume-87_slice_497.jpg
maskImg = cv2.imread(path_test_mask + "/" + maskFile, 0)
#maskImg = cv2.imread("./test_masks_tumor/" + maskFile,0)  # volume-87_slice_497_mask.png (imread(fname, mode), mode: 0 for grayscale, 1 color, -1 unchanged, nothing: brings all three in array)
predMaskImg = cv2.imread(path_pred_mask + "/" + predMaskFile, 0)
#predMaskImg = cv2.imread("./pred_masks_tumor_from_liver_masked/" + predMaskFile, 0) #volume-87_slice_497_pred_mask.png

if predMaskImg is None: # if no pred image found means prediction was a full of zeroes matrix, no livenr, no tumor
    predMaskImg = np.zeros(imgDim)
if maskImg is None: # if no pred image found means prediction was a full of zeroes matrix, no livenr, no tumor
    maskImg = np.zeros(imgDim)
 
#%% resize image prediction mask (128x128) to fit to originals (they are 512x512) 
predMaskImg = cv2.resize(predMaskImg, imgDim, interpolation = cv2.INTER_AREA)

#convert and calculate 1s and 2s
nppredMaskImg=np.array(predMaskImg)
npmaskImg=np.array(maskImg)
npmaskImg1s = np.count_nonzero(npmaskImg == 1)
nppredMaskImg1s = np.count_nonzero(nppredMaskImg == 1)
print("npmaskImg=1: ", npmaskImg1s)
print("nppredMaskImg=1: " , nppredMaskImg1s)
#maskImg = Image.fromarray((255.0 / a.max() * (a - a.min())).astype(np.uint8), mode="L") # to show dif between 1 and 2 of liver and tumor, good to see image, to save it I want 1 and two values to compare with ground truth
#maskImg = Image.fromarray((a*40).astype(np.uint8), mode="L")     # not good       
nppredMaskImgLight=nppredMaskImg*120 # multiply it to make it brighter and visible to plot
npmaskImgLight=npmaskImg*120

print(np.unique(npmaskImg, return_counts=True))
print(np.unique(nppredMaskImg, return_counts=True))

metrics = get_metrics(npmaskImg, nppredMaskImg)

print('Metrics: ')
print(metrics)


plot_sample_pred([liverImg, npmaskImgLight,nppredMaskImgLight]) #plot ct, mask and pred_mask, one on another
