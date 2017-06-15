import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import signal as sg
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import matplotlib.pyplot as plt

# from scipy.signal import convolve2d as conv2

from skimage import color, data, restoration


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
    #a = np.array([np.sin(x) for i in range(40)])
    a=1-image(4)
    grille = np.zeros(shape=(31,31))
    for i in range(7):
        grille[i*5]=1
        grille[:,i*5]=1
    #a=ndimage.imread("../src/sin100.png")[:,:,0]
    conv=sg.convolve2d(b,a)
    #conv+=0.1 * conv.std() * np.random.standard_normal(conv.shape)
    deconvolved_RL= np.array(restoration.wiener(conv, a,0.000000000005,clip=False))
    #print(deconvolved_RL)
    #deconvolved_RL= np.array(restoration.unsupervised_wiener(conv, a))
    """0.000005 marche très bien"""
    mid=np.array(deconvolved_RL.shape)//2
    # print(mid)
    # plt.subplot(411)
    # plt.imshow(b)
    # plt.subplot(412)
    # plt.imshow(a)
    # plt.subplot(413)
    # plt.imshow(conv)
    # plt.subplot(414)
    # plt.imshow(deconvolved_RL[mid[0]-mid2[0]:mid[0]+mid2[0],mid[1]-mid2[1]:mid[1]+mid2[1]])


    plt.subplot(211)
    plt.imshow(b)
    plt.subplot(212)
    plt.imshow(deconvolved_RL[mid[0]-mid2[0]:mid[0]+mid2[0],mid[1]-mid2[1]:mid[1]+mid2[1]])



    plt.show()


class App(Frame):
    def __init__(self, master):
        # Create a container
        Frame.__init__(self,master)

        # Create Buttons
        self.nameim = Label(self, text='déconv')
        self.nameim.pack(side='left')
        self.case = Entry(self, width=5)
        self.case.pack(side='left')
        self.namegrille = Label(self, text='avec')
        self.namegrille.pack(side='left')
        self.case0 = Entry(self, width=5)
        self.case0.pack(side='left')
        self.dec = Button(self, text="Do", command=self.deconv)
        self.dec.pack(side='left')

        self.name = Label(self, text='balance')
        self.name.pack(side='left')
        self.case1 = Entry(self, width=5)
        self.case1.pack(side='left')
        self.new = Button(self, text="add", command=self.otherbalance)
        self.new.pack(side='left')

        self.name2 = Label(self, text='name')
        self.name2.pack(side='left')
        self.case2 = Entry(self, width=5)
        self.case2.pack(side='left')
        self.new2 = Button(self, text="Save", command=self.savefig)
        self.new2.pack(side='left')

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)

        x = np.array(np.linspace(-10, 10, 40))
        y = np.array(np.linspace(-10, 10, 100))


        self.b = np.array([np.sin(i * y) for i in range(100)])
        self.a = np.array([np.sin(x) for i in range(40)])
        self.conv = sg.convolve2d(self.b, self.a)
        self.deconvolved_RL = np.array(restoration.wiener(self.conv, self.a, 0.000005, clip=False))
        self.mid2 = np.array(self.b.shape) // 2
        self.mid = np.array(self.deconvolved_RL.shape) // 2
        self.line = self.ax.imshow(self.deconvolved_RL[self.mid[0]-self.mid2[0]:self.mid[0]+self.mid2[0],self.mid[1]-self.mid2[1]:self.mid[1]+self.mid2[1]])

        self.canvas = FigureCanvasTkAgg(self.fig,master=master)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        Frame.pack(self)

        Button(self, text="Quit", command=quit).pack()
        #Button(self, text="Save", command=plt.savefig(str(self.case2.get())+".png")).pack()

    def otherbalance(self):
        self.ax.clear()
        entry=str(self.case1.get())
        balance=float(entry)
        self.deconvolved_RL = np.array(restoration.wiener(self.conv, self.a, balance=balance, clip=False))
        self.line = self.ax.imshow(self.deconvolved_RL[self.mid[0]-self.mid2[0]:self.mid[0]+self.mid2[0],self.mid[1]-self.mid2[1]:self.mid[1]+self.mid2[1]])
        self.canvas.draw()

    def savefig(self):
        #self.fig.savefig("/home/petitcolas/PycharmProjects/IPCMS/src/" + str(self.case2.get()))
        self.fig.savefig("./" + str(self.case2.get()))

    def deconv(self):
        self.ax.clear()
        data=ndimage.imread("../src/"+str(self.case.get()))[:,:,0]
        grille=ndimage.imread("../src/"+str(self.case0.get()))[:,:,0]
        conv=sg.convolve2d(data, grille)
        self.deconvolved_RL=np.array(restoration.wiener(conv, grille, 1100, clip=False))
        self.mid2 = np.array(data.shape) // 2
        self.mid = np.array(self.deconvolved_RL.shape) // 2

        self.line = self.ax.imshow(self.deconvolved_RL[self.mid[0]-self.mid2[0]:self.mid[0]+self.mid2[0],self.mid[1]-self.mid2[1]:self.mid[1]+self.mid2[1]])
        self.canvas.draw()

def FFt():
    x = np.array(np.linspace(-50, 50, 400))
    a = np.array([np.cos(x) for i in range(400)])
    #img=ndimage.imread("../src/170608af(100nm_20x_OlympusNA1).bmp")[:, :, 0]
    #img=ndimage.imread("/home/petitcolas/PycharmProjects/IPCMS/src/170608af(100nm_20x_OlympusNA1).bmp")[:, :, 0]
    #img=ndimage.imread("/home/petitcolas/PycharmProjects/IPCMS/src/lena_256_NB.png")#[:, :, 0]
    #img=ndimage.imread("/home/petitcolas/PycharmProjects/IPCMS/src/170608ac(10µm_grille_20x_OlympusNA1).bmp")[:, :, 0]
    #img=ndimage.imread("/home/petitcolas/PycharmProjects/IPCMS/src/170608aa(200nm_20x_OlympusNA1).bmp")[:, :, 0]

    #img=ndimage.rotate(a,45,reshape=False,order=2)[100:300,100:300]
    #print(img.shape)
    # grille = np.zeros(shape=(301,301))
    # for i in range(100):
    #     grille[i*3]=1
    #     grille[:,i*3]=1
    # img=grille
    img=a

    """magnitude spectrum"""
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    # img2 = 20 * np.log(np.abs(fshift))
    # f2 = np.fft.fft2(img2)
    # fshift2 = np.fft.fftshift(f2)
    magnitude_spectrum = 20 * np.log(1+np.abs(fshift))

    #plt.subplot(121), plt.imshow(img, cmap='gray')
    #plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122),
    plt.imshow(magnitude_spectrum, cmap='jet')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


    """phase + amplification"""
    # fft=np.fft.fft2(img)
    # real=np.real(fft)
    # imag=np.imag(fft)
    # amplitude=np.sqrt(np.multiply(real,real)+np.multiply(imag,imag))
    # phase=np.arctan(np.divide(imag,real))
    # print(phase.shape)
    # print(amplitude.shape)
    # sig=amplitude*np.exp(phase*1j)
    # plt.subplot(211)
    # plt.imshow(amplitude)
    # plt.subplot(212)
    # plt.imshow(phase)
    # plt.show()

FFt()
# bla=ndimage.imread("../src/gamma_fin1.gif")[:,:,0]
# grille=ndimage.imread("../src/grille.gif")[:,:,0]
# conv=sg.convolve2d(bla, grille)
# deconvolved_RL=np.array(restoration.wiener(conv, grille, 0.0000000000005, clip=False))
# mid2 = np.array(bla.shape) // 2
# mid = np.array(deconvolved_RL.shape) // 2
#
# plt.imshow(deconvolved_RL[mid[0]-mid2[0]:mid[0]+mid2[0],mid[1]-mid2[1]:mid[1]+mid2[1]])
# plt.show()
# root = Tk()
# App(root)
# root.mainloop()
#grille2()
#test()
# grille = np.zeros(shape=(31,31))
# for i in range(7):
#     grille[i*5]=1
#     grille[:,i*5]=1
#
# plt.imshow(1-grille)
# plt.show()
