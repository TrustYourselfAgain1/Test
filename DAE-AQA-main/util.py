# -*- coding: utf-8 -*-


import torch
import logging
import math
import torch.nn as nn
import numpy as np

alpha, beta = torch.Tensor([0.6]).cuda(), torch.Tensor([0.4]).cuda()


# 创建一个日志记录器（logger）对象，它将日志信息保存到指定的文件路径中
def get_logger(filepath, log_info):
    logger = logging.getLogger(filepath)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)  # 创建一个文件处理器，用于将日志信息写入到文件中
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)  # 将文件处理器添加到日志记录器中，确保日志信息会被写入到文件中
    logger.info('-' * 30 + log_info + '-' * 30)
    return logger  # 函数返回创建好的日志记录器对象


def log_and_print(logger, msg):
    logger.info(msg)
    print(msg)


def loss_function_v2(sigma, x, mu):
    sigma = torch.Tensor(sigma).cuda()
    x = torch.Tensor(x).cuda()
    mu = torch.Tensor(mu).cuda()
    MSE_loss = nn.MSELoss(reduction='sum')
    rec_loss = alpha/(sigma)*MSE_loss(x, mu)
    sup_loss = beta*torch.log(sigma)
    return rec_loss+sup_loss


def loss_function(recon_x, x, mu):
    MSE_loss = nn.MSELoss(reduction='sum')
    reconstruction_loss = MSE_loss(recon_x, x)+MSE_loss(x, mu)
    return reconstruction_loss

