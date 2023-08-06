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
path = Path(".")
path_im = path/'train_images' #train_images_liver
path_lbl = path/'train_masks' #train_masks_liver

fnames = get_image_files(path_im)
lbl_names = get_image_files(path_lbl)

#get_msk = lambda o: path/'train_masks_liver'/f'{o.stem}_mask{o.suffix}'
get_msk = lambda o: path/'train_masks'/f'{o.stem}_mask.png'
codes = np.array(["background","liver", "tumor"]) #np.array(["background","liver","tumor"])


def half(sz):
    return tuple(int(x/2) for x in sz)

sz = (512, 512)#msk.shape;
half = tuple(int(x/2) for x in sz); half
quoter = tuple(int(x/4) for x in sz); quoter #to train with quoter image size 128

camvid = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                   get_items=get_image_files,
                   splitter= RandomSplitter(valid_pct=0.15, seed=42), #valid_pct=0.15 seed for reproducability of results in random split data for validation
                   get_y=get_msk,
                   batch_tfms=[*aug_transforms(size=half), Normalize.from_stats(*imagenet_stats)])

dls = camvid.dataloaders(path/'train_images', bs=8) #train_images_liver
dls.vocab = codes
name2id = {v:k for k,v in enumerate(codes)}

void_code = name2id['background']

def acc_camvid(inp, targ):#not working with fine tune
  targ = targ.squeeze(1)
  mask = targ != void_code
  return (inp.argmax(dim=1)[mask]==targ[mask]).float().mean()

def foreground_acc(inp, targ, bkg_idx=0, axis=1):  # exclude a background from metric
    "Computes non-background accuracy for multiclass segmentation"
    targ = targ.squeeze(1)
    mask = targ != bkg_idx
    return (inp.argmax(dim=axis)[mask]==targ[mask]).float().mean() 

def cust_foreground_acc(inp, targ):  # include a background into the metric
    return foreground_acc(inp=inp, targ=targ, bkg_idx=3, axis=1) # 3 is a dummy value to include the background which is 0


#pt = ranger
learn = unet_learner(dls, resnet50, metrics= [foreground_acc, cust_foreground_acc], self_attention=True, act_cls=Mish, opt_func = ranger) # best

learn.fine_tune(40, wd=0.1, cbs=SaveModelCallback() )  #60 best


learn.save('Liv_n_tum_segm_resnet50_40ep_CrossEntropyLossFlat') ####////////////////////////////////////////////////////////////////////////////////////////////
#learn1 = learn.load('Liver_segmentation_only_fastai')
learn.show_results(max_n=4, figsize=(18,8))
#test
dl = learn.dls.test_dl(fnames[:5])
dl.show_batch()
preds = learn.get_preds(dl=dl)
pred_1 = preds[0][0]
print(pred_1.shape)


