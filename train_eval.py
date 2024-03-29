#!/usr/bin/env python
#-*- coding: utf-8 -*-
import os
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
from torch.optim import lr_scheduler

from opt import opt
from data import Data
from network import MGN
from loss import Loss
from utils import get_optimizer,extract_feature,CenterLoss
from metrics import mean_ap, cmc, re_ranking

import logging
import logging.handlers
import scipy.io

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Main():
    def __init__(self, model, loss, data):
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        self.testset = data.testset
        self.queryset = data.queryset

        self.model = model.to('cuda')
        # self.model=nn.DataParallel(model).cuda()
        self.loss = loss
        self.optimizer = get_optimizer(model)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=0.1)
        ###############################
        self.center_criterion=CenterLoss(numberclasses=751,feat_dim=256,use_gpu=True)
        self.optimizer_center=torch.optim.SGD(self.center_criterion.parameters(),lr=0.5)

    def train(self):

        self.scheduler.step()

        self.model.train()
        for batch, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            # loss.backward()
            # self.optimizer.step()
            centerloss=[self.center_criterion(outputs,labels) for output in outputs[1:5]]
            Center_Loss=sum(centerloss)/len(centerloss)
            loss_sum=loss+0.0005*Center_Loss

            self.optimizer_center.zero_grad()
            loss_sum.backward()

            self.optimizer.step()
            for param in self.center_criterion.parameters():
                param.grad.data*=(1./0.0005)
            self.optimizer_center.step()


    def evaluate(self,logger):

        self.model.eval()

        print('extract features, this may take a few minutes')
        qf = extract_feature(self.model, tqdm(self.query_loader)).numpy()
        gf = extract_feature(self.model, tqdm(self.test_loader)).numpy()

        def rank(dist):
            r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

            return r, m_ap

        #########################   re rank##########################
        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

        r, m_ap = rank(dist)

        print('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))

        #########################no re rank##########################
        dist = cdist(qf, gf)

        r, m_ap = rank(dist)

        print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))

    def vis(self):

        self.model.eval()

        gallery_path = data.testset.imgs
        gallery_label = data.testset.ids

        # Extract feature
        print('extract features, this may take a few minutes')
        query_feature = extract_feature(model, tqdm([(torch.unsqueeze(data.query_image, 0), 1)]))
        gallery_feature = extract_feature(model, tqdm(data.test_loader))

        # sort images
        query_feature = query_feature.view(-1, 1)
        score = torch.mm(gallery_feature, query_feature)
        score = score.squeeze(1).cpu()
        score = score.numpy()

        index = np.argsort(score)  # from small to large
        index = index[::-1]  # from large to small

        # # Remove junk images
        # junk_index = np.argwhere(gallery_label == -1)
        # mask = np.in1d(index, junk_index, invert=True)
        # index = index[mask]

        # Visualize the rank result
        fig = plt.figure(figsize=(16, 4))

        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        plt.imshow(plt.imread(opt.query_image))
        ax.set_title('query')

        print('Top 10 images are as follow:')

        for i in range(10):
            img_path = gallery_path[index[i]]
            print(img_path)

            ax = plt.subplot(1, 11, i + 2)
            ax.axis('off')
            plt.imshow(plt.imread(img_path))
            ax.set_title(img_path.split('/')[-1][:9])

        fig.savefig("show.png")
        print('result saved to show.png')

def create_logger():
    global logger
    logger=logging.getLogger('Logger')
    logger.setLevel(logging.DEBUG)
    handler=logging.handlers.RotatingFileHandler('./test.log',maxBytes=0,backupCount=2000)
    formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

if __name__ == '__main__':
    create_logger()
    logger.info('BEGIN')
    data = Data()
    model = MGN()
    loss = Loss()
    main = Main(model, loss, data)

    if opt.mode == 'train':

        for epoch in range(1, opt.epoch + 1):
            print('epoch', epoch)
            main.train()
            if epoch % 50 == 0:
                print('start evaluate')
                main.evaluate(logger)
                # os.makedirs('weights', exist_ok=True)
                # torch.save(model.state_dict(), ('weights/model_{}.pt'.format(epoch)))
                PATH_CHECKP='weights/checkpoint_epoch'+str(epoch)+'.pth.tar'
                torch.save({
                    'epoch':epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':main.optimizer.state_dict(),
                    'scheduler_dict':main.scheduler.state_dict(),
                    'optimizer_center':main.optimizer_center.state_dict(),
                })

    if opt.mode == 'evaluate':
        print('start evaluate')
        model.load_state_dict(torch.load(opt.weight))
        main.evaluate()
    if opt.mode=='resume':
        print(' start resuming')
        checkpoint=torch.load('checkpoint_epoch100.pth.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        main.scheduler.load_state_dict(checkpoint['scheduler_dict'])
        main.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        main.optimizer_center.load_state_dict(checkpoint['optimizer_center'])
        epoch=checkpoint['epoch']

        for epoch in range(epoch, opt.epoch + 1):
            print('epoch', epoch)
            main.train()
            if epoch % 50 == 0:
                print('start evaluate')
                main.evaluate(logger)
                # os.makedirs('weights', exist_ok=True)
                # torch.save(model.state_dict(), ('weights/model_{}.pt'.format(epoch)))
                PATH_CHECKP='weights/checkpoint_epoch'+str(epoch)+'.pth.tar'
                torch.save({
                    'epoch':epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':main.optimizer.state_dict(),
                    'scheduler_dict':main.scheduler.state_dict(),
                    'optimizer_center':main.optimizer_center.state_dict(),
                })




    if opt.mode == 'vis':
        print('visualize')
        model.load_state_dict(torch.load(opt.weight))
        main.vis()