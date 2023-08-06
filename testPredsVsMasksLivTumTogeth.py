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
    
def get_std_metrics(TP, TN, FP, FN):
    accuracy = (TP + TN) / (TP+FP+TN+FN) # for two dimension: divide by /rows*cols to get total items, for one dim len(y_true) is enough
    assert not np.isnan(TN)
    assert not np.isnan(FP)
    assert not np.isnan(FN)
    assert not np.isnan(TP)
    recall = precision = f1_score = 0
    if TP + FP > 0:
        precision = TP / (TP + FP)
    if TP + FP > 0:
        recall = TP / (TP + FN)
    if precision + recall > 0:
        f1_score = 2 * ((precision * recall) / (precision + recall))
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    } 
# %% Model testing, one for test, final value:  (0, len(df_files))
#Unet outputs mask 128 image size (we resize also inputs to 128 to proccess them)
#originals are 512
#by scaling predictions to 512 I can compare
#but count pixels doesn t give metric for accuracy (stil 128 pred image haw less pixels)
# maybe if train without resize give better results but complexity, load;;; results look good event with resize
#the resize I do bellow to ipred mask image to become 512 from 128 probably decreases accuracy but .....


#folders to check
path_test_im = './test_images' #3053
path_test_mask = './test_masks'
path_pred_mask = './pred_masks_Liv_n_tum_segm_resnet50_40ep_CrossEntropyLossFlat' #pred_masks_livntum ///////////////////////////////////////////////////////////////////////////////////////

#f = open('results.csv',mode = 'w') s.replace('xyz', '')
f = open(f"results-{path_pred_mask.replace('./', '')}.csv",mode = 'w') 
f.write('liverFile, maskImg1s, maskImg2s, predMaskImg1s, predMaskImg2s, LivAccuracy, lTP, lTN, lFP, lFN, lprecision, lrecall, lf1_score, liou, lDSC, lFNR, lFPR, lSens, lSpec')#firstRow headings
f.write(', tumAccuracy, tTP, tTN, tFP, tFN, tprecision, trecall, tf1_score, tiou, tDSC, tFNR, tFPR, tSens, tSpec, maskLiverContours, maskTumorContours, predMaskLiverContours, predMaskTumorContours  \n')

conf_matrix_liver =  np.zeros((2,2),dtype = int) #to create confusion matrix to see performance for liver n tumor detection per sample based on pixel instance
conf_matrix_tum =  np.zeros((2,2),dtype = int)
conf_matrix_liverC =  np.zeros((2,2),dtype = int) #to create confusion matrix to see performance for liver n  tumor detection per sample based on contours
conf_matrix_tumC =  np.zeros((2,2),dtype = int)

imgDim = (512, 512) #dim mask Images should be compared, 512 original sizes, convert pred 128-->512


#%%loop start
counter=0
file_iterator = os.scandir(path_test_im)
for i in file_iterator:
    liverImg = cv2.imread(path_test_im + "/" + i.name,-1)
    liverFile = i.name
    #liverFile = "volume-98_slice_616.jpg"#"volume-98_slice_616.jpg" volume-92_slice_452
    maskFile = liverFile.split(".")[0] + "_mask.png"
    #if pred mask file not exists it means  it was full of zeros, it predicted background only:
    predMaskFile = liverFile.split(".")[0] + "_mask.png"
    #liverImg = cv2.imread("./test_images/" + liverFile)  # volume-87_slice_497.jpg
    maskImg = cv2.imread(path_test_mask + "/" +maskFile,0)  # volume-87_slice_497_mask.png (imread(fname, mode), mode: 0 for grayscale, 1 color, -1 unchanged, nothing: brings all three in array)
    predMaskImg = cv2.imread(path_pred_mask + "/" + predMaskFile, 0) #volume-87_slice_497_pred_mask.png

    if predMaskImg is None: # if no pred image found means prediction was a full of zeroes matrix, no livenr, no tumor
        predMaskImg = np.zeros(imgDim)
   
    #convert and calculate 1s and 2s
    nppredMaskImg=np.array(predMaskImg)
    nppredMaskImg = cv2.resize(nppredMaskImg.astype(float), imgDim, interpolation = cv2.INTER_AREA)
    npmaskImg=np.array(maskImg)
    npmaskImg1s = np.count_nonzero(npmaskImg == 1)
    npmaskImg2s = np.count_nonzero(npmaskImg == 2)
    nppredMaskImg1s = np.count_nonzero(nppredMaskImg == 1)
    nppredMaskImg2s = np.count_nonzero(nppredMaskImg == 2)
    

    liverMetric = get_metrics_liver(npmaskImg, nppredMaskImg)
    tumMetrics = get_metrics_tum(npmaskImg, nppredMaskImg)

   
    #%%check contours for exiatance of liver or tumor
    #check tumor-liver by contours in test mask
    maskOnlyBackground = predMaskOnlyBackground = maskLiverContours = maskTumorContours = predMaskLiverContours=  predMaskTumorContours = 0
    if np.count_nonzero(maskImg)==0: #["background","liver","tumor"] if all zero we have only background
            maskOnlyBackground =1
    else:
        maskOnlyBackground =0
        # for liver
        Img_liver = np.where(maskImg == 1, 255, maskImg) # if 1 liver, make 255, else get value of img, img_liver is the liver mask
        Img_liver = np.where(Img_liver == 2, 0, maskImg) # if 2 tumor, remove it, we want only if found liver
        ret = check(Img_liver) # check image for liver contours, if find, next, write 1 in second bit for liver found
        if ret: 
            maskLiverContours =1
        else:
            maskLiverContours =0
        # for tumor
        img_tumor = np.where(maskImg == 2, 255, maskImg) # if 2 tumor, make 255, img_tumor will contain the liver mask in the end
        img_tumor = np.where(img_tumor == 1, 0, maskImg) # if 1 liver, remove it (0), we want only if found tumor
        ret = check(img_tumor)  # check image for tumor contours, if find, next, write 1 in third bit for tumor found
        if ret:
            maskTumorContours =1
        else:
            maskTumorContours =0

    #chek tumor-liver by contours in prediction  mask, same as above, to compare
    if np.count_nonzero(predMaskImg)==0: #["background","liver","tumor"] if all zero we have only background
            predMaskOnlyBackground =1
    else:
        predMaskOnlyBackground =0
        # for liver
        Img_liver = np.where(predMaskImg == 1, 255, predMaskImg) # if 1 liver, make 255, else get value of img, img_liver is the liver mask
        Img_liver = np.where(Img_liver == 2, 0, predMaskImg) # if 2 tumor, remove it, we want only if found liver
        ret = check(Img_liver) # check image for liver contours, if find, next, write 1 in second bit for liver found
        if ret: 
            predMaskLiverContours =1
        else:
            predMaskLiverContours =0
        # for tumor
        img_tumor = np.where(predMaskImg == 2, 255, predMaskImg) # if 2 tumor, make 255, img_tumor will contain the liver mask in the end
        img_tumor = np.where(img_tumor == 1, 0, predMaskImg) # if 1 liver, remove it (0), we want only if found tumor
        ret = check(img_tumor)  # check image for tumor contours, if find, next, write 1 in third bit for tumor found
        if ret:
            predMaskTumorContours =1
        else:
            predMaskTumorContours =0


    f.write(liverFile + ', ' + str(npmaskImg1s) + ', ' + str(npmaskImg2s) + ', ' + str(nppredMaskImg1s) + ', ' + str(nppredMaskImg2s) + ', ' + str(liverMetric["accuracy"]) +', ' + str(liverMetric["TP"]) +', ' + str(liverMetric["TN"])+', ' + str(liverMetric["FP"])+', ' + str(liverMetric["FN"])+ ', ' + str(liverMetric["precision"])+', ' + str(liverMetric["recall"]) +', ' + str(liverMetric["f1_score"])+', ' + str(liverMetric["iou"])+', ' + str(liverMetric["DSC"])+', ' +  str(liverMetric["FNR"])+', ' +  str(liverMetric["FPR"])+', ' + str(liverMetric["Sens"])+', ' + str(liverMetric["Spec"]))
    f.write(', ' + str(tumMetrics["accuracy"]) +', ' + str(tumMetrics["TP"]) +', ' + str(tumMetrics["TN"])+', ' + str(tumMetrics["FP"])+', ' + str(tumMetrics["FN"])+', ' + str(tumMetrics["precision"])+', ' + str(tumMetrics["recall"]) +', ' + str(tumMetrics["f1_score"])+', ' + str(tumMetrics["iou"]) + ', ' + str(tumMetrics["DSC"])+', ' + str(tumMetrics["FNR"])+', ' +  str(tumMetrics["FPR"]) + ', ' + str(tumMetrics["Sens"])+', ' + str(tumMetrics["Spec"]) + ', ' + str(maskLiverContours) + ', ' + str(maskTumorContours) + ', ' + str(predMaskLiverContours) + ', ' + str(predMaskTumorContours) +'\n')

    #conf matrix tum by tumor pixel existance 
    if nppredMaskImg2s > 0  and npmaskImg2s > 0:
        conf_matrix_tum[0,0] += 1
    if nppredMaskImg2s == 0 and npmaskImg2s == 0:
        conf_matrix_tum[1,1] += 1
    if nppredMaskImg2s == 0 and npmaskImg2s > 0:
        conf_matrix_tum[1,0] += 1
    if nppredMaskImg2s > 0 and npmaskImg2s == 0:
        conf_matrix_tum[0,1] += 1

    #conf matrix liverby liver pixel existance
    if nppredMaskImg1s > 0  and npmaskImg1s > 0:
        conf_matrix_liver[0,0] += 1
    if nppredMaskImg1s == 0 and npmaskImg1s == 0:
        conf_matrix_liver[1,1] += 1
    if nppredMaskImg1s == 0 and npmaskImg1s > 0:
        conf_matrix_liver[1,0] += 1
    if nppredMaskImg1s > 0 and npmaskImg1s == 0:
        conf_matrix_liver[0,1] += 1

    #conf matrix tum by tumor conturs existance
    if predMaskTumorContours > 0  and maskTumorContours > 0:
        conf_matrix_tumC[0,0] += 1
    if predMaskTumorContours == 0 and maskTumorContours == 0:
        conf_matrix_tumC[1,1] += 1
    if predMaskTumorContours == 0 and maskTumorContours > 0:
        conf_matrix_tumC[1,0] += 1
    if predMaskTumorContours > 0 and maskTumorContours == 0:
        conf_matrix_tumC[0,1] += 1

    #conf matrix liverby liver contours existance
    if predMaskLiverContours > 0  and maskLiverContours > 0:
        conf_matrix_liverC[0,0] += 1
    if predMaskLiverContours == 0 and maskLiverContours == 0:
        conf_matrix_liverC[1,1] += 1
    if predMaskLiverContours == 0 and maskLiverContours > 0:
        conf_matrix_liverC[1,0] += 1
    if predMaskLiverContours > 0 and maskLiverContours == 0:
        conf_matrix_liverC[0,1] += 1


# metrics concerning whole image, if tum and liver correctly detected
lTP = conf_matrix_liver[0,0]
lFP = conf_matrix_liver[0,1]
lFN = conf_matrix_liver[1,0]
lTN = conf_matrix_liver[1,1]
std_metrics_liver = get_std_metrics(lTP, lTN, lFP, lFN)

tTP = conf_matrix_tum[0,0]
tFP = conf_matrix_tum[0,1]
tFN = conf_matrix_tum[1,0]
tTN = conf_matrix_tum[1,1]
std_metrics_tum = get_std_metrics(tTP, tTN, tFP, tFN)

f.write('\n')    
f.write('Image Level Ddetection Liver totaly: , lTP, lFP, lFN, lTN, LivAccuracy, lprecision, lrecall, lf1_score \n')
f.write(' ,'+ str(lTP) + ', ' + str(lFP)  + ', '  + str(lFN) + ', ' + str(lTN) +  ', ' + str(std_metrics_liver["accuracy"]) + ', ' + str(std_metrics_liver["precision"]) +', ' + str(std_metrics_liver["recall"]) +', ' + str(std_metrics_liver["f1_score"]) +'\n') 
f.write('Image Level Ddetection tumor totaly: , tTP, tFP, tFN, tTN, tumAccuracy, tprecision, trecall, tf1_score \n')
f.write(' ,'+ str(tTP) + ', ' + str(tFP)  + ', '  + str(tFN) + ', ' + str(tTN) +  ', ' + str(std_metrics_tum["accuracy"]) + ', ' + str(std_metrics_tum["precision"]) +', ' + str(std_metrics_tum["recall"]) +', ' + str(std_metrics_tum["f1_score"]) +'\n') 
 
#%%loop end
    
# print(conf_matrix_liver)
# print(conf_matrix_tum)
#Plot Confusion Matrix tum
ax = sns.heatmap(conf_matrix_tum, annot=True, cmap='Blues')
ax.set_title('Confusion Matrix with labels for tumor detection by pixel existance');
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values ')
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['True','False'])
ax.yaxis.set_ticklabels(['True','False'])
## Display the visualization of the Confusion Matrix.
fig = plt.gcf()
plt.show(block=True)
fig.savefig('conf_matrix_tum.png')

plt.close()
ax1 = sns.heatmap(conf_matrix_liver, annot=True, cmap='Blues')
ax1.set_title('Confusion Matrix with labels for liver detection by pixel existance')
ax1.set_xlabel('Actual Values')
ax1.set_ylabel('Predicted Values ')
## Ticket labels - List must be in alphabetical order
ax1.xaxis.set_ticklabels(['True','False'])
ax1.yaxis.set_ticklabels(['True','False'])
fig = plt.gcf()
plt.show(block=True)
fig.savefig('conf_matrix_liver.png')

plt.close()
#Plot Confusion Matrix tumc
ax2 = sns.heatmap(conf_matrix_tumC, annot=True, cmap='Blues')
ax2.set_title('Confusion Matrix with labels for tumor detection by contours');
ax2.set_xlabel('Actual Values')
ax2.set_ylabel('Predicted Values ')
## Ticket labels - List must be in alphabetical order
ax2.xaxis.set_ticklabels(['True','False'])
ax2.yaxis.set_ticklabels(['True','False'])
## Display the visualization of the Confusion Matrix.
fig = plt.gcf()
plt.show(block=True)
fig.savefig('conf_matrix_tumC.png')

plt.close()
ax3 = sns.heatmap(conf_matrix_liverC, annot=True, cmap='Blues')
ax3.set_title('Confusion Matrix with labels for liver detection by contours')
ax3.set_xlabel('Actual Values')
ax3.set_ylabel('Predicted Values ')
## Ticket labels - List must be in alphabetical order
ax3.xaxis.set_ticklabels(['True','False'])
ax3.yaxis.set_ticklabels(['True','False'])
fig = plt.gcf()
plt.show(block=True)
fig.savefig('conf_matrix_liverC.png')

plt.close()

f.close()


