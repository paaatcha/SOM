import numpy as np
import matplotlib.pyplot as plt
from som import SOM

# Training inputs for RGBcolors
colors = np.array(
    [[0., 0., 0.],
     [0., 0., 1.],
     [0., 0., 0.5],
     [0.125, 0.529, 1.0],
     [0.33, 0.4, 0.67],
     [0.6, 0.5, 1.0],
     [0., 1., 0.],
     [1., 0., 0.],
     [0., 1., 1.],
     [1., 0., 1.],
     [1., 1., 0.],
     [1., 1., 1.],
     [.33, .33, .33],
     [.5, .5, .5],
     [.66, .66, .66]])

colors2 = np.array(
    [[0., 0., 0.],
     [0., 0., 1.],
     [1., 1., 0.],
     [1., 1., 1.],
     [1., 0., 0.]])

color_names = \
    ['black', 'blue', 'darkblue', 'skyblue',
     'greyblue', 'lilac', 'green', 'red',
     'cyan', 'violet', 'yellow', 'white',
     'darkgrey', 'mediumgrey', 'lightgrey']

s = SOM(colors2, [25, 25], alpha=0.3)

# Initial weights
plt.imshow(s.w_nodes)
plt.show()

# Learning to cluster the RGB colors
s.train(max_it=30)

# Trained weights
plt.imshow(s.w_nodes)
plt.show()
