import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from config import config

class SiamFCNet(nn.Module):
    def __init__(self, training=True):
        super(SiamFCNet,self).__init__()
        self.feature_map = nn.Sequential(
            nn.Conv2d(3,96,11,2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2),
            nn.Conv2d(96,256,5,1,groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1,groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256,3,1,groups=2)
        )
        self.bais = nn.Parameter(torch.zeros(1))
        self.exemplar = None
        self.training = training

        if self.training:
            #generate label
            gt, weight = self.create_label((config.response_sz, config.response_sz))
            # Get labels and convert it to GPU
            self.train_gt = torch.from_numpy(gt).cuda()
            self.train_weight = torch.from_numpy(weight).cuda()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, z, x):

        if z is not None and x is not None:
            feature_z = self.feature_map(z)
            feature_x = self.feature_map(x)
            # print(feature_z.shape)
            # print(feature_x.shape)
            N = feature_x.shape[0]
            feature_x = feature_x.view(1, -1, feature_x.shape[2], feature_x.shape[3])
            score = F.conv2d(feature_x, feature_z, groups=N)*config.response_scale + self.bais
            score = score.view(N,-1,score.shape[2],score.shape[3])
            return score
        elif z is None and x is not None:
            feature_x = self.feature_map(x)
            N = feature_x.shape[0]
            feature_x = feature_x.view(1, -1, feature_x.shape[2], feature_x.shape[3])
            score = F.conv2d(feature_x, self.exemplar, groups=N) * config.response_scale + self.bais
            score = score.view(N, -1, score.shape[2], score.shape[3])
            return score
        else:
            self.exemplar = self.feature_map(z)
            self.exemplar = torch.cat([self.exemplar for _ in range(3)], dim=0)



    def loss(self, pred):

        # return F.binary_cross_entropy_with_logits(pred, self.train_gt,
        #             self.train_weight, reduction='mean')  # normalize the batch_size
        cost_function = nn.BCEWithLogitsLoss(weight=self.train_weight, reduction='sum')
        loss = cost_function(pred, self.train_gt) #/config.train_batch_size
        return loss

    def create_label(self, shape):
        # same for all pairs
        h, w = shape
        y = np.arange(h, dtype=np.float32) - (h - 1) / 2.
        x = np.arange(w, dtype=np.float32) - (w - 1) / 2.
        y, x = np.meshgrid(y, x)
        dist = np.sqrt(x ** 2 + y ** 2)
        label = np.zeros((h, w))
        label[dist <= config.radius / config.total_stride] = 1
        label = label[np.newaxis, :, :]
        weights = np.ones_like(label)
        weights[label == 1] = 0.5 / np.sum(label == 1)
        weights[label == 0] = 0.5 / np.sum(label == 0)
        label = np.repeat(label, config.train_batch_size, axis=0)[:, np.newaxis, :, :]
        return label.astype(np.float32), weights.astype(np.float32)

