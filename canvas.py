# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, 2018年 12月 21日 星期五 21:35:14 CST
# ***
# ************************************************************************************/

import visdom
import math
import numpy as np


class Canvas(object):
    """Canvas Object."""

    def __init__(self):
        """Canvas."""
        self.env = "torch"
        self.name = "canvas"
        try:
            self.vis = visdom.Visdom(env=self.env, raise_exceptions=True)
        except:
            self.vis = None
            print("Could not connect to visdom server, please make sure:")
            print("1. install visdom:")
            print("   pip insall visdom or conda install visdom")
            print("2. start visdom server: ")
            print("   python -m visdom.server &")

    def setname(self, name):
        """Change canvas name."""
        self.name = name

    def draw_tensor(self, tensor, description="tensor description"):
        """Draw Tensor with BxCxHxW dimension. B,C,H,W --> B*C, 1, H, W."""
        assert tensor.dim() == 4

        if self.vis:
            x = tensor.clone().cpu()
            B, C, H, W = x.size()
            if C != 1 and C != 3:
                x = x.view(B * C, 1, H, W)
                n = math.ceil(math.sqrt(B * C))
            else:
                n = math.ceil(math.sqrt(B))
            if n > 1280 / W:
                n = int(1280 / W)
            a = x.min()
            b = x.max()
            x.add_(-a).div_(b - a + 1e-5)
            title = description + ": {}x{}x{}x{}".format(B, C, H, W)
            self.vis.images(x, nrow=n, win=self.name, env=self.env, opts=dict(title=title))
        else:
            print("Visdom server seems not running.")

    def draw_loss_acc(self, epoch, loss, acc):
        """Draw loss and accuray"""
        if self.vis is None:
            print("Visdom server seems not running.")
            return

        if epoch == 0:
            self.vis.line(
                X=np.array([0]),
                Y=np.column_stack((np.array([0]), np.array([0]))),
                win=self.name,
                opts=dict(
                    title=self.name + ' loss & acc',
                    legend=['loss', 'acc'],
                    width=1280,
                    height=720,
                    xlabel='Epoch',
                ))
        else:
            self.vis.line(
                X=np.array([epoch]),
                Y=np.column_stack((np.array([loss]), np.array([acc]))),
                win=self.name,
                update='append')


def tensor_show(tensor, window, description=""):
    c = Canvas()
    c.setname(window)
    c.draw_tensor(tensor, description)
