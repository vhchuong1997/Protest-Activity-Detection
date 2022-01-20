"""
created by: Donghyeon Won
"""

from __future__ import print_function
import os
import argparse
import numpy as np
import pandas as pd
import time
import shutil
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.models as models

from util import ProtestDatasetEval, modified_resnet50 , modified_EffnetB1, modified_EffnetB2, modified_EffnetB3


def eval_one_dir(img_dir, model,img_size, img_size2):
        """
        return model output of all the images in a directory
        """
        model.eval()
        # make dataloader
        dataset = ProtestDatasetEval(img_dir,img_size, img_size2)
        data_loader = DataLoader(dataset,
                                num_workers = args.workers,
                                batch_size = args.batch_size)
        # load model

        outputs = []
        imgpaths = []

        n_imgs = len(os.listdir(img_dir))
        with tqdm(total=n_imgs) as pbar:
            for i, sample in enumerate(data_loader):
                imgpath, input = sample['imgpath'], sample['image']
                if args.cuda:
                    input = input.cuda()

                input_var = Variable(input)
                output = model(input_var)
                outputs.append(output.cpu().data.numpy())
                imgpaths += imgpath
                if i < n_imgs / args.batch_size:
                    pbar.update(args.batch_size)
                else:
                    pbar.update(n_imgs%args.batch_size)


        df = pd.DataFrame(np.zeros((len(os.listdir(img_dir)), 13)))
        df.columns = ["imgpath", "protest", "violence", "sign", "photo",
                      "fire", "police", "children", "group_20", "group_100",
                      "flag", "night", "shouting"]
        df['imgpath'] = imgpaths
        df.iloc[:,1:] = np.concatenate(outputs)
        df.sort_values(by = 'imgpath', inplace=True)
        return df

def main():

    # load trained model
    print("*** loading model from {model}".format(model = args.model))
    # model = modified_resnet50()
    model = modified_EffnetB3()
    if args.model == 'EffNetB1':
        model = modified_EffnetB1()
    elif args.model == 'ResNet':
        model = modified_resnet50()
    elif args.model == 'EffNetB2':
        model = modified_EffnetB2()
    elif args.model == 'EffNetB3':
        model = modified_EffnetB3()
    
    if args.model == 'EffNetB1':
        img_size = (256, 240)
    elif args.model == 'ResNet':
        img_size = (224, 224)
    elif args.model == 'EffNetB2':
        img_size = (288, 288)
    elif args.model == 'EffNetB3':
        img_size = (320, 300)
    imgsizelist = list(img_size)
    imgsizelist[0] = round(imgsizelist[0]*1.1)
    imgsizelist[1] = round(imgsizelist[1]*1.1)    
    img_size2 = tuple(imgsizelist)

    if args.cuda:
        model = model.cuda()
    with open(args.model_dir,'rb') as f:
        model.load_state_dict(torch.load(f)['state_dict'])
    print("*** calculating the model output of the images in {img_dir}"
            .format(img_dir = args.img_dir))

    # calculate output
    df = eval_one_dir(args.img_dir, model, img_size, img_size2)

    # write csv file
    df.to_csv(args.output_csvpath, index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir",
                        type=str,
                        required = True,
                        help = "image directory to calculate output"
                        "(the directory must contain only image files)"
                        )
    parser.add_argument("--output_csvpath",
                        type=str,
                        default = "result.csv",
                        help = "path to output csv file"
                        )
    # parser.add_argument("--model",
    #                     type=str,
    #                     required = True,
    #                     help = "model path"
    #                     )
    parser.add_argument("--cuda",
                        action = "store_true",
                        help = "use cuda?",
                        )
    parser.add_argument("--workers",
                        type = int,
                        default = 4,
                        help = "number of workers",
                        )
    parser.add_argument("--batch_size",
                        type = int,
                        default = 16,
                        help = "batch size",
                        )
    parser.add_argument("--model_dir",
                        type=str,
                        default = 'model_best.pth.tar',
                        help = "directory path to save model file",
                        )
    parser.add_argument("--model",
                        type=str,
                        default = 'ResNet',
                        help = "name of the model",
                        )
    args = parser.parse_args()

    main()
