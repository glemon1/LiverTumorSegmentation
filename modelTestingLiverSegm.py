from fastai.vision.all import *

from fastcore.xtras import Path

from fastai.callback.hook import summary
from fastai.callback.progress import ProgressCallback
from fastai.callback.schedule import lr_find, fit_flat_cos

from fastai.data.block import DataBlock
from fastai.data.external import untar_data, URLs
from fastai.data.transforms import get_image_files, FuncSplitter, Normalize

from fastai.layers import Mish
from fastai.losses import BaseLoss
from fastai.optimizer import ranger

from fastai.torch_core import tensor

from fastai.vision.augment import aug_transforms
from fastai.vision.core import PILImage, PILMask
from fastai.vision.data import ImageBlock, MaskBlock, imagenet_stats
from fastai.vision.learner import unet_learner

from PIL import Image
import numpy as np

from torch import nn
from torchvision.models.resnet import resnet34

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import glob

import cv2
import imageio

path = Path(".")

#assert (Path(path)/'models/Liver_segmentation_only_fastai.pth').exists()  
#Test load did load the model
codes = np.array(["background","liver"])
get_msk = lambda o: path/'train_masks_liver_all'/f'{o.stem}_mask.png'

sz = (512, 512)#msk.shape;
half = tuple(int(x/2) for x in sz)
camvid = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                   get_items=get_image_files,
                   splitter= RandomSplitter(valid_pct=0.15, seed=42),#seed same as preproc
                   get_y=get_msk,
                   batch_tfms=[*aug_transforms(size=half), Normalize.from_stats(*imagenet_stats)])
                   
dls = camvid.dataloaders(path/'train_images', bs=8)

learn0 = unet_learner(dls, resnet50, self_attention=True, act_cls=Mish, opt_func=ranger) # this is the original

#learn0 = unet_learner(resnet34)
learn0 = learn0.load('Liver_segmentation_resnet50ep40_CrossEntropy_diceMetrics')####################################################################################################

def show_mask(preds):#show masks 0-1 nicely
    for i, pred in enumerate(preds[0]): #to save preds
        pred_arg = pred.argmax(dim=0).numpy()
        rescaled = (255.0 / pred_arg.max() * (pred_arg - pred_arg.min())).astype(np.uint8)
        im = Image.fromarray(rescaled)
        im.save(f'Image_{i}.png')

conf_matrix =  np.zeros((2,2),dtype = int)

# %% Model testing, one for test, final value:  (0, len(df_files))
SAVE_MASK_PRED_FILES= True

imgDim = (512, 512) #dim mask Images should be compared, 512 original sizes, convert pred 128-->512

#%%loop start
path_test_im = './test_images'
test_im_fnames = get_image_files(path_test_im)
test_imgs_num = len(test_im_fnames)
batch_size  = 5 # no of images to procces in learn0.dls.test_dl(test_im_fnames[:4]), test_im_fnames must be scalar, if only one does not work
#TypeError: 'PosixPath' object is not subscriptable
loop_rerpeats =  test_imgs_num // batch_size # floor division to find loop  nums
print(test_imgs_num)
print(loop_rerpeats)

print(test_im_fnames[:15])
print(type(test_im_fnames))


counter=0
for i in range(loop_rerpeats):
    if ((i+1)*batch_size < test_imgs_num -1):   
        batch_end = (i+1)*batch_size
        batch_items = batch_size
    else:
        batch_end = (test_imgs_num - 1)
        batch_items = batch_size % batch_size # in last loop avoid popinter out of borders in test_im_fnames
    dl = learn0.dls.test_dl(test_im_fnames[i*batch_size:batch_end])
    #learn0.show_results(max_n=2, figsize=(12,6))
    #dl.show_batch()
    preds = learn0.get_preds(dl=dl)
    for pred_no in range(batch_items):
        pred = preds[0][pred_no]
        print('learn output image shape:')
        print(pred.shape)
        predicted_mask = pred.argmax(dim=0)#convert from 3 layer image to 1 layer mask
        a=np.array(predicted_mask)
        
        testLiverFile = str(test_im_fnames[i*batch_size + pred_no]) # filename
        testMaskFile = str(testLiverFile).split("/")[1]
        testMaskFile = testMaskFile.split(".")[0] + "_mask.png"
        testMaskImg = cv2.imread("./test_masks_liver/" + testMaskFile, 0)  #test masks with liver
        if testMaskImg is None: # if no pred image found means prediction was a full of zeroes matrix, no livenr, no tumor
            testMaskImg = np.zeros(imgDim)     
        
        if (np.any(a) and SAVE_MASK_PRED_FILES): # print/save masks with liver only else no save zero mask, 
            #%% resize image prediction mask (256x256) to fit to originals (they are 512x512) 
            np_pred_mask_img = cv2.resize(a.astype(float), imgDim, interpolation = cv2.INTER_AREA)
            npPredMaskImg1s = np.count_nonzero(np_pred_mask_img == 1) #liver pxels count in prediction (1s)
            pred_mask_img = Image.fromarray(np_pred_mask_img.astype(np.uint8)) # to save mask as it is (better)
            print('save image shape (after resize)')
            print(pred_mask_img.shape)
            pred_mask_img.save(f"pred_masks_liver_segmentation_resnet50ep40_CrossEntropy_diceMetrics/{testMaskFile}")###############################################################################
        
        #convert and calculate 1s 
        npTestMaskImg=np.array(testMaskImg)
        npTestMaskImg1s = np.count_nonzero(npTestMaskImg == 1)
        npPredMaskImg1s = np.count_nonzero(a == 1)  
        print(testLiverFile)
        print('test-pred:')
        print(npTestMaskImg1s)
        print(npPredMaskImg1s)

        if npPredMaskImg1s > 0:
            liver_p = True
        else:
            liver_p = False

        # for geting the actual mask values
        
        if npTestMaskImg1s > 0:
            liver_t = True
        else:
            liver_t = False
            
        # populating the conf_matrix
        if liver_p == True and liver_t == True:
            conf_matrix[0,0] += 1
        if liver_p == False and liver_t == False:
            conf_matrix[1,1] += 1
        if liver_p == False and liver_t == True:
            conf_matrix[1,0] += 1
        if liver_p == True and liver_t == False:
            conf_matrix[0,1] += 1

    
print(conf_matrix)

#Plot Confusion Matrix
import seaborn as sns

ax = sns.heatmap(conf_matrix, annot=True, cmap='Blues')

ax.set_title('Liver segmentation results \n');
ax.set_xlabel('\nActual Values for liver')
ax.set_ylabel('Predicted Values for liver existence');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['True','False'])
ax.yaxis.set_ticklabels(['True','False'])

## Display the visualization of the Confusion Matrix.
plt.show()

