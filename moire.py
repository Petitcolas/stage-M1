import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import signal as sg

import matplotlib.pyplot as plt

# from scipy.signal import convolve2d as conv2

from skimage import color, data, restoration


def showMatrix(Z, filename=None):
    """trace une matrice en niveau de violet"""
    """si on rentre un vecteur, cela le transforme en matrice"""
    Z = np.atleast_2d(Z)

    plt.figure(figsize=(Z.shape[1]/2.,Z.shape[0]/2.), dpi=72)
    plt.imshow(Z, cmap='jet', extent=[0,Z.shape[1],0,Z.shape[0]],vmin=0, vmax=max(1,Z.max()), interpolation='nearest', origin='upper')
    plt.xticks([]), plt.yticks([])
    plt.xlim(0,Z.shape[1])
    plt.ylim(0,Z.shape[0])

    if filename is not None:plt.savefig(filename,dpi=72)
    plt.show()

def image(i):
    if i==0:
        return ndimage.imread("../src/gamma_fin1.gif")[:,:,0]
    elif i==1:
        return ndimage.imread("../src/cosmos_petit.jpg")[:, :, 0]
    elif i==2:
        return ndimage.imread("../src/cosmos_moyen.jpg")[:, :, 0]
    elif i==3:
        return ndimage.imread("../src/cosmos_grand.jpg")[:, :, 0]
    elif i==4:
        return ndimage.imread("../src/grille.gif")[:, :, 0]
    elif i==5:
        return ndimage.imread("../src/patate.jpeg")[:, :, 0]
    elif i==6:
        return ndimage.imread("../src/patate.jpg")[:, :, 0]
    elif i==7:
        return ndimage.imread("../src/points.png")[:, :, 0]

def test():
    x = np.array(np.linspace(-10, 10, 40))
    y = np.array(np.linspace(-10, 10, 100))
    #a = np.ones((5, 5)) / 25
    b = np.array([np.sin(i * y) for i in range(100)])
    # a= np.zeros(shape=(31,31))
    # a[15,15]=1
    # b=image(1)
    mid2=np.array(b.shape)//2
    a = np.array([np.sin(x) for i in range(40)])
    conv=sg.convolve2d(b,a)
    fourier=sg.convolution2d(b,a)


    plt.subplot(411)
    plt.imshow(b)
    plt.subplot(412)
    plt.imshow(a)
    plt.subplot(413)
    plt.imshow(conv)
    plt.subplot(414)
    plt.imshow(fourier)
    plt.show()


def grille1():


    grille = np.zeros(shape=(31,31))
    #grille=image(4)[0:256,0:200]
    # #grille[15,15]=1
    for i in range(7):
        grille[i*5]=1
        grille[:,i*5]=1

    img1=image(0)
    conv=sg.convolve(img1, grille)
    #conv+=0.1 * conv.std() * np.random.standard_normal(conv.shape)
    # conv_noise=conv.copy()
    # conv_noise+=(np.random.poisson(lam=25, size=conv.shape) - 10) / 255.
    deconvolved_RL = restoration.wiener(conv,grille,1100)
    plt.subplot(411)
    plt.imshow(img1)
    plt.subplot(412)
    plt.imshow(grille)
    plt.subplot(413)
    plt.imshow(conv)
    plt.subplot(414)
    plt.imshow(np.log(deconvolved_RL))
    plt.show()

def grille2():
    x = np.array(np.linspace(-10, 10, 40))
    y = np.array(np.linspace(-10, 10, 100))
    #a = np.ones((5, 5)) / 25
    #b = np.array([np.sin(i * y) for i in range(100)])
    # a= np.zeros(shape=(31,31))
    # a[15,15]=1
    b=image(3)
    mid2=np.array(b.shape)//2
    a = np.array([np.sin(x) for i in range(40)])
    conv=sg.convolve2d(b,a)
    #conv+=0.1 * conv.std() * np.random.standard_normal(conv.shape)
    deconvolved_RL= np.array(restoration.wiener(conv, a,0.000005,clip=False))
    print(deconvolved_RL)
    #deconvolved_RL= np.array(restoration.unsupervised_wiener(conv, a))
    """0.000005 marche très bien"""
    mid=np.array(deconvolved_RL.shape)//2
    # print(mid)
    plt.subplot(211)
    plt.imshow(b)
    # plt.subplot(412)
    # plt.imshow(a)
    # plt.subplot(413)
    # plt.imshow(conv)
    plt.subplot(212)
    plt.imshow(deconvolved_RL[mid[0]-mid2[0]:mid[0]+mid2[0],mid[1]-mid2[1]:mid[1]+mid2[1]])
    plt.show()

grille2()
#test()
