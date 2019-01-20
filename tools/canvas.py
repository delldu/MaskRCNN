# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, 2018年 12月 21日 星期五 21:35:14 CST
# ***
# ************************************************************************************/

import visdom
import math
import numpy as np
import torchvision
from PIL import Image
import matplotlib.pyplot as plt


def plt_loss_acc(name, loss, acc):
    """Draw loss and accuray via matplotlib."""
    loss = np.array(loss)
    acc = np.array(acc)

    plt.figure(name)
    plt.gcf().clear()
    plt.plot(loss, label='loss')
    plt.plot(acc, label='acc')
    plt.xlabel('Epoch')
    plt.ylabel('Loss & Acc')
    plt.legend()
    plt.show(block=False)
    plt.pause(0.1)


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
            print("Warning: visdom server is not working, please make sure:")
            print("1. install visdom:")
            print("   pip insall visdom or conda install visdom")
            print("2. start visdom server: ")
            print("   python -m visdom.server &")
            self.loss = []
            self.acc = []

    def setname(self, name):
        """Change canvas name."""
        self.name = name

    def draw_tensor(self, tensor, description="tensor description"):
        """Draw Tensor with BxCxHxW dimension. B,C,H,W --> B*C, 1, H, W."""
        assert tensor.dim() == 4

        x = tensor.clone()
        B, C, H, W = x.size()
        if C != 1 and C != 3:
            x = x.view(B * C, 1, H, W)
            n = math.ceil(math.sqrt(B * C))
        else:
            n = math.ceil(math.sqrt(B))
        if n > 1280 / W:
            n = max(1, int(1280 / W))
        a = x.min()
        if a < 0:
            x.add(-a)
        b = x.max()
        if (b > 1.0):
            x.div_(b + 1e-5)
        x = x.cpu()

        if self.vis:
            title = description + ": {}x{}x{}x{}".format(B, C, H, W)
            self.vis.images(x, nrow=n, win=self.name, env=self.env, opts=dict(title=title))
        else:
            grid = torchvision.utils.make_grid(x, nrow=n)
            ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            image = Image.fromarray(ndarr)

            image.show()

    def draw_loss_acc(self, epoch, loss, acc):
        """Draw loss and accuray."""
        if self.vis is None:
            # Visdom server not available.
            self.loss.append(loss)
            self.acc.appen(acc)
            plt_loss_acc(self.name, self.loss, self.acc)
            return

        if epoch == 0:
            self.vis.line(
                X=np.array([0]),
                Y=np.column_stack((np.array([0]), np.array([0]))),
                win=self.name,
                opts=dict(
                    title=self.name,
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


def tensor_show(tensor, window="canvas", description=""):
    """Show tensor."""
    c = Canvas()
    c.setname(window)
    c.draw_tensor(tensor, description)
