import os

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def main_print(log) -> None:
    if ('LOCAL_RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0:
        print(log)


def main_tqdm(inner):
    if ('LOCAL_RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0:
        return tqdm(inner)
    else:
        return inner


"""
nerf.py의 novel_pose plot 위한 코드 
"""


def scatter_point(data1,data2=None,data3=None):
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(data1[:,0],data1[:,1],data1[:,2],color='r',alpha = 0.5)
    if data2 is not None :
        ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2],color='g' ,alpha=0.5)
    if data3 is not None :
        ax.scatter(data3[:, 0], data3[:, 1], data3[:, 2],color='b' ,alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def plot_line(data1, data2=None):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')

    N = data1.shape[0]
    for i in range(N):
        ax.plot([0,data1[i,0]],[0,data1[i,1]], [0,data1[i,2]], color='r', linewidth=0.5, alpha=0.5)

    if data2 is not None:
        N = data2.shape[0]
        for i in range(N):
            ax.plot([0, data2[i, 0]], [0, data2[i, 1]], [0, data2[i, 2]], color='b', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    # png_fname = "{}/{}.png".format(path,ep)
    # plt.savefig(png_fname,dpi=75)
    # clean up
    # plt.clf()
