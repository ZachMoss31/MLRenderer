''' Image Preprocessing::
Theory
    - images can be viewed as a function()... grayscale is a function of f(x,y) of pixel location,
        which maps -> a certain grey level like an int from [0,255] or float from [0,1].
        * RGB image can be seen as fr(x,y), fg(x,y), and fb(x,y), which maps RGB values respectively.
Goal
    - load image into memory via some data structure, and save it back to disk post-modificaiton

Tools
    - numpy, scipy, scikit-image, opencv-python, matplotlib

'''

# using scikit-image's imread() as our method to read image into memory from disc,
#   use numpy for array operations, and matplotlib for displaying
''' is this the entry point we can use to display nerual images instead, sub matplotlib?'''

from skimage.io import imread
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Plot pixel values for a given channel
def plot_3d(X,Y,Z, cmap='Reds', title='') -> None:
    fig = plt.figure(15,15)

