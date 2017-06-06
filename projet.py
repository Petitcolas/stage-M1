from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons,RectangleSelector
import numpy as np
name=["141215ac.dat","150707ax.dat"]


"""On charge le .dat ss forme de np"""
def load(i):
    imname="../src/"+name[i]
    Im=open(imname)
    return np.loadtxt(imname)


"""On calcule le nombre d'image ds le .dat"""
def extract(name):
    I=load(name)
    K = max(np.shape(I))
    r = 1
    for q in range(0, (K - 1), 1):
        if I[q + 1, 0] - I[q, 0] != 0:
            r = r + 1


    N = r  # N nombre d'images N de l'acquisition

    H = min(np.shape(I)) - 1  # H nombre de colonnes (retranchée de la colonne retard)
    V = max(np.shape(I)) / N  # V nombre de lignes

    Pas = abs(I[int(V), 0] - I[int(V) - 1, 0])  # Calcul le pas de l'acquisition (en ps)
    t = np.empty(N)
    for i in range(N):
        t[i]= Pas*i


    """Le np ou se trouvent ttes les img"""
    V = max(np.shape(I)) / N  # V nombre de lignes dans une image

    Pic = np.zeros((int(N), int(V), int(H)))
    for q in range(0, N, 1):  # N nombre d'images
        Pic[q, :, :] = I[ int(V) * q : int(V) * (q + 1), 1:]
    return Pic




"""coupe longitudinale"""


def coupe(name):
    Pic = extract(name)
    return np.rollaxis(np.rollaxis(Pic, 1, 0), 2, 0)



"""graph d'un rectangle avec médiane ou moyenne à choisir"""


def graph2D(Pic):
    taille = Pic.shape[0]
    Val = np.zeros(shape=taille)
    Val2 = np.zeros(shape=taille)
    for i in range(taille):
        Val[i] = -np.mean(Pic[i])
        """on peut utiliser la moyenne ou la mediane il faut voire avec les chercheurs de l'ipcms ce qu'ils
        préfèrent"""
        Val2[i] = -np.median(Pic[i])
    return np.arange(Pic.shape[0]),Val

def show(val):

    Pic = extract(val)

    N = Pic.shape[0]
    fig = plt.figure()
    ax = fig.add_subplot(122)
    fig.subplots_adjust(left=0.25, bottom=0.25)

    """définition des colorsmap possibles"""
    def image(i,val):
        ax.clear()
        if i==1:
            return ax.imshow(Pic[int(np.round(val))], cmap='Greys')
        elif i==2:
            return ax.imshow(Pic[int(np.round(val))], cmap='Greys_r')
        else:
            return ax.imshow(Pic[int(np.round(val))], cmap='jet')
    image(1,0)
    #fig.colorbar(im)

    """initialisation de la selection rectangulaire"""
    def line_select_callback(eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        ay = fig.add_subplot(121)
        ay.clear()
        ay.plot(graph2D(Pic[:,int(x1):int(x2),int(y1):int(y2)])[0],
                graph2D(Pic[:,int(x1):int(x2),int(y1):int(y2)])[1])
        plt.ylabel('intensité lumineuse')
        plt.xlabel('profondeur de l\'image')
        plt.draw()

    def toggle_selector(event):
        print(' Key pressed.')
        if event.key in ['Q', 'q'] and toggle_selector.RS.active:
            print(' RectangleSelector deactivated.')
            toggle_selector.RS.set_active(False)
        if event.key in ['A', 'a'] and not toggle_selector.RS.active:
            print(' RectangleSelector activated.')
            toggle_selector.RS.set_active(True)

    axcolor = 'lightgoldenrodyellow'
    axstyle = fig.add_axes([0.05, 0.7, 0.15, 0.15], axisbg=axcolor)



    """initialisation bouton changement de cmap"""
    radio = RadioButtons(axstyle, ('Greys', 'Greys_r', 'jet'))
    styldct = {'Greys': 1, 'Greys_r': 2, 'jet': 3}


    def style(label):
        image(styldct[label],stemps.val)
        plt.draw()

    radio.on_clicked(style)

    """initialisation slider"""
    def update(val):
        image(styldct[radio.value_selected],val)
        fig.canvas.draw()
    axtemps = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
    stemps = Slider(axtemps, 'profondeur de l\' image', 0, N - 1, valinit=0, valfmt='%1.0f')

    stemps.on_changed(update)
    toggle_selector.RS = RectangleSelector(ax, line_select_callback,
                                           drawtype='box', useblit=True,
                                           button=[1, 3],  # don't use middle button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    plt.connect('key_press_event', toggle_selector)


    plt.show()


#show(coupe(name2))
show(0)


