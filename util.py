"""
created by: Donghyeon Won
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models


class ProtestDataset(Dataset):
    """
    dataset for training and evaluation
    """
    def __init__(self, txt_file, img_dir, transform = None):
        """
        Args:
            txt_file: Path to txt file with annotation
            img_dir: Directory with images
            transform: Optional transform to be applied on a sample.
        """
        self.label_frame = pd.read_csv(txt_file, delimiter="\t").replace('-', 0)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.label_frame)
    def __getitem__(self, idx):
        imgpath = os.path.join(self.img_dir,
                                self.label_frame.iloc[idx, 0])
        image = pil_loader(imgpath)

        # protest = self.label_frame.iloc[idx, 1:2].as_matrix().astype('float')
        # violence = self.label_frame.iloc[idx, 2:3].as_matrix().astype('float')
        # visattr = self.label_frame.iloc[idx, 3:].as_matrix().astype('float')
        protest = torch.tensor(self.label_frame.iloc[idx, 1], dtype=torch.float)
        # violence = torch.tensor(self.label_frame.iloc[idx, 2:3], dtype=torch.float)
        violence = torch.tensor(np.asarray(self.label_frame.iloc[idx, 2]).astype('float'),dtype=torch.float)

        visattr = torch.tensor(np.asarray(self.label_frame.iloc[idx, 3:]).astype('float'), dtype=torch.float)
        # torch.tensor(self.label_frame.iloc[idx, 3:].astype('float'))
        label = {'protest':protest, 'violence':violence, 'visattr':visattr}

        sample = {"image":image, "label":label}
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        return sample

class ProtestDatasetEval(Dataset):
    """
    dataset for just calculating the output (does not need an annotation file)
    """
    def __init__(self, img_dir,img_size, img_size2):
        """
        Args:
            img_dir: Directory with images
        """
        self.img_dir = img_dir
        self.transform = transforms.Compose([
                                transforms.Resize(img_size),
                                transforms.CenterCrop(img_size2),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                                ])
        self.img_list = sorted(os.listdir(img_dir))
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        imgpath = os.path.join(self.img_dir,
                                self.img_list[idx])
        image = pil_loader(imgpath)
        # we need this variable to check if the image is protest or not)
        sample = {"imgpath":imgpath, "image":image}
        sample["image"] = self.transform(sample["image"])
        return sample

class FinalLayer(nn.Module):
    """modified last layer for resnet50 for our dataset"""
    def __init__(self):
        super(FinalLayer, self).__init__()
        self.fc = nn.Linear(2048, 12)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)
        return out

class FinalLayer2(nn.Module):
    """modified last layer for resnet50 for our dataset"""
    def __init__(self):
        super(FinalLayer2, self).__init__()
        self.fc = nn.Linear(1280, 12)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)
        return out

class FinalLayer3(nn.Module):
    """modified last layer for resnet50 for our dataset"""
    def __init__(self):
        super(FinalLayer3, self).__init__()
        self.fc = nn.Linear(1408, 12)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)
        return out

class FinalLayer4(nn.Module):
    """modified last layer for resnet50 for our dataset"""
    def __init__(self):
        super(FinalLayer4, self).__init__()
        self.fc = nn.Linear(1536, 12)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)
        return out

class Effnet_b1(nn.Module):
    def __init__(self):
        super(Effnet_b1, self).__init__()
        self.eff =  models.efficientnet_b1(pretrained = True)
        for param in self.eff.parameters():
            param.requires_grad = False
        self.l1 = nn.Linear(1000 , 256)
        self.dropout = nn.Dropout(0.25)
        self.l2 = nn.Linear(256,12)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.eff(input)
        # x = x.view(x.size(0),-1)
        x = self.dropout(self.relu(self.l1(x)))
        x = self.l2(x)
        x = self.sigmoid(x)
        return x

class Effnet_b2(nn.Module):
    def __init__(self):
        super(Effnet_b2, self).__init__()
        self.eff =  models.efficientnet_b2(pretrained = True)
        for param in self.eff.parameters():
            param.requires_grad = False
        self.l1 = nn.Linear(1000 , 256)
        self.dropout = nn.Dropout(0.25)
        self.l2 = nn.Linear(256,12)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.eff(input)
        # x = x.view(x.size(0),-1)
        x = self.dropout(self.relu(self.l1(x)))
        x = self.l2(x)
        x = self.sigmoid(x)
        return x

class Effnet_b3(nn.Module):
    def __init__(self):
        super(Effnet_b3, self).__init__()
        self.eff =  models.efficientnet_b3(pretrained = True)
        for param in self.eff.parameters():
            param.requires_grad = False
        self.l1 = nn.Linear(1000 , 256)
        self.dropout = nn.Dropout(0.25)
        self.l2 = nn.Linear(256,12)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.eff(input)
        # x = x.view(x.size(0),-1)
        x = self.dropout(self.relu(self.l1(x)))
        x = self.l2(x)
        x = self.sigmoid(x)
        return x

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def modified_resnet50():
    # load pretrained resnet50 with a modified last fully connected layer
    model = models.resnet50(pretrained = True)
    model.fc = FinalLayer()

    # uncomment following lines if you wnat to freeze early layers
    # i = 0
    # for child in model.children():
    #     i += 1
    #     if i < 4:
    #         for param in child.parameters():
    #             param.requires_grad = False


    return model

def modified_EffnetB1():
    # load pretrained resnet50 with a modified last fully connected layer
    '''
    model = models.efficientnet_b1(pretrained = True)
    model.classifier = FinalLayer2()
    '''
    model = Effnet_b1()
    
    # uncomment following lines if you wnat to freeze early layers
    # i = 0
    # for child in model.children():
    #     i += 1
    #     if i < 4:
    #         for param in child.parameters():
    #             param.requires_grad = False


    return model

def modified_EffnetB2():
    # load pretrained resnet50 with a modified last fully connected layer
    '''
    model = models.efficientnet_b2(pretrained = True)
    model.classifier = FinalLayer3()
    '''
    model = Effnet_b2()
    return model

def modified_EffnetB3():
    # load pretrained resnet50 with a modified last fully connected layer
    '''
    model = models.efficientnet_b3(pretrained = True)
    model.classifier = FinalLayer4()
    '''
    model = Effnet_b3()
    return model

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count != 0:
            self.avg = self.sum / self.count

class Lighting(object):
    """
    Lighting noise(AlexNet - style PCA - based noise)
    https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/experiments/recognition/dataset/minc.py
    """
    
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))
