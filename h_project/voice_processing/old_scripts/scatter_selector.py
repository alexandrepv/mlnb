
# From: https://stackoverflow.com/questions/7969352/matplotlib-interactively-select-points-or-locations
from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.image import AxesImage
import numpy as np
from numpy.random import rand


fig, ax = plt.subplots()
ax.set_title('click on points', picker=True)
ax.set_ylabel('ylabel', picker=True, bbox=dict(facecolor='red'))
line, = ax.plot(rand(100), 'o', picker=5)

def onpick1(event):
    if isinstance(event.artist, Line2D):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        print('X=' + str(np.take(xdata, ind)[0]))  # Print X point
        print('Y=' + str(np.take(ydata, ind)[0]))  # Print Y point

fig.canvas.mpl_connect('pick_event', onpick1)

plt.show()
