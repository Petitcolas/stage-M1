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

def grille1():


    #grille = np.zeros(shape=(31,31))
    grille=np.ones((5, 5)) / 25
    # #grille[15,15]=1
    # for i in range(7):
    #     grille[i*5]=1
    #     grille[:,i*5]=1

    img1=ndimage.imread("gamma_fin1.gif")[:,:,0]
    conv=sg.convolve2d(img1, grille,"same")
    conv+=0.1 * conv.std() * np.random.standard_normal(conv.shape)
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
    plt.imshow(deconvolved_RL)
    plt.show()

def grille2():
    x = np.array(np.linspace(-10, 10, 40))
    y = np.array(np.linspace(-10, 10, 100))
    #a = np.ones((5, 5)) / 25
    b = np.array([np.sin(i * y) for i in range(100)])
    a = np.array([np.sin(x) for i in range(40)])
    conv=sg.convolve(b,a)
    #conv+=0.1 * conv.std() * np.random.standard_normal(conv.shape)
    deconvolved_RL= np.array(restoration.wiener(conv, a,0.005))
    mid=np.array(deconvolved_RL.shape)//2
    print(mid)
    plt.subplot(211)
    plt.imshow(b)
    # plt.subplot(412)
    # plt.imshow(a)
    # plt.subplot(413)
    # plt.imshow(conv)
    plt.subplot(212)
    plt.imshow(deconvolved_RL[mid[0]-50:mid[0]+50,mid[1]-50:mid[1]+50])
    plt.show()

grille2()

# astro = color.rgb2gray(data.astronaut())
#
# psf = np.ones((5, 5)) / 25
# #astro = conv2(astro, psf, 'same')
# # Add Noise to Image
# astro_noisy = astro.copy()
# astro_noisy += (np.random.poisson(lam=25, size=astro.shape) - 10) / 255.
#
# # Restore Image using Richardson-Lucy algorithm
# deconvolved_RL = restoration.richardson_lucy(astro_noisy, psf, iterations=30)
#
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
# plt.gray()
#
# for a in (ax[0], ax[1], ax[2]):
#        a.axis('off')
#
# ax[0].imshow(astro)
# ax[0].set_title('Original Data')
#
# ax[1].imshow(astro_noisy)
# ax[1].set_title('Noisy data')
#
# ax[2].imshow(deconvolved_RL, vmin=astro_noisy.min(), vmax=astro_noisy.max())
# ax[2].set_title('Restoration using\nRichardson-Lucy')
#
#
# fig.subplots_adjust(wspace=0.02, hspace=0.2,
#                     top=0.9, bottom=0.05, left=0, right=1)
# plt.show()
