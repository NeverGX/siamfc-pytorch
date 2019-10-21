import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import argparse
import torchvision
import numpy as np
import os
import cv2
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import config
from siamfc import SiamFCNet
from dataset import GOT_10KDataset
from custom_transforms import Normalize, ToTensor, RandomStretch,RandomCrop, CenterCrop, RandomBlur, ColorAug
import sys

torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_dir',type=str, default='/home/wangkh/Downloads/got-10k/crop_train_data', help='got_10k train dir')
parser.add_argument('--val_data_dir',type=str, default='/home/wangkh/Downloads/got-10k/crop_val_data', help='got_10k val dir')
arg = parser.parse_args()
train_data_dir = arg.train_data_dir
val_data_dir = arg.val_data_dir

def main():


    train_z_transforms = transforms.Compose([
        RandomStretch(),
        CenterCrop((config.exemplar_size, config.exemplar_size)),
        ToTensor()
    ])
    train_x_transforms = transforms.Compose([
        RandomStretch(),
        RandomCrop((config.instance_size, config.instance_size),
                   config.max_translate),
        ToTensor()
    ])
    val_z_transforms = transforms.Compose([
        CenterCrop((config.exemplar_size, config.exemplar_size)),
        ToTensor()
    ])
    val_x_transforms = transforms.Compose([
        ToTensor()
    ])

    # create dataset
    train_dataset = GOT_10KDataset(train_data_dir, train_z_transforms, train_x_transforms)
    valid_dataset = GOT_10KDataset(val_data_dir, val_z_transforms, val_x_transforms, training=False)

    trainloader = DataLoader(train_dataset, batch_size=config.train_batch_size,
                             shuffle=True, pin_memory=True, num_workers=config.train_num_workers, drop_last=True)
    validloader = DataLoader(valid_dataset, batch_size=config.valid_batch_size,
                             shuffle=False, pin_memory=True, num_workers=config.valid_num_workers, drop_last=True)
    # create summary writer
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    summary_writer = SummaryWriter(config.log_dir)

    # start training
    with torch.cuda.device(config.gpu_id):
        model = SiamFCNet()
        model.init_weights()
        # model.load_state_dict(torch.load('./models/siamfc_30.pth'))
        model = model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,
                                    momentum=config.momentum, weight_decay=config.weight_decay)
        # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        schdeuler = StepLR(optimizer, step_size=config.step_size,
                           gamma=config.gamma)

        for epoch in range(config.epoch):
            train_loss = []
            model.train()
            for i, data in enumerate(tqdm(trainloader)):
                z, x = data
                z, x = Variable(z.cuda()), Variable(x.cuda())
                outputs = model(z, x)
                optimizer.zero_grad()
                loss = model.loss(outputs)
                loss.backward()
                optimizer.step()
                step = epoch * len(trainloader) + i
                summary_writer.add_scalar('train/loss', loss.data, step)
                train_loss.append(loss.data)
            train_loss = np.mean(train_loss)
            valid_loss = []
            model.eval()
            for i, data in enumerate(tqdm(validloader)):
                z, x = data
                z, x = Variable(z.cuda()), Variable(x.cuda())
                outputs = model(z, x)
                loss = model.loss(outputs)
                valid_loss.append(loss.data)
            valid_loss = np.mean(valid_loss)
            print("EPOCH %d valid_loss: %.4f, train_loss: %.4f, learning_rate: %.4f" %
                  (epoch, valid_loss, train_loss, optimizer.param_groups[0]["lr"]))
            summary_writer.add_scalar('valid/loss',
                                      valid_loss, epoch + 1)
            torch.save(model.cpu().state_dict(),
                       "./models/siamfc_{}.pth".format(epoch + 1))
            model.cuda()
            schdeuler.step()

if __name__ == '__main__':
    sys.exit(main())


