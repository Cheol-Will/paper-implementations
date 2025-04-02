import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def tensorboard_write(writer, n_iter, correct, test_loss, avg_loss):
    writer.add_scalar(f"Acc", correct, n_iter)
    writer.add_scalar(f"Loss", test_loss, n_iter)
    writer.add_scalar(f"Train_Loss", avg_loss, n_iter)