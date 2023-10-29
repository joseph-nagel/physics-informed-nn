'''Visualization tools.'''

import numpy as np
import matplotlib.pyplot as plt


def make_colors(num_colors,
                qual_cm=plt.cm.Dark2,
                seq_cm=plt.cm.viridis,
                ensure_seq=False):
    '''
    Create different colors.

    Parameters
    ----------
    num_colors : int
        Number of colors to create.
    qual_cm : matplotlib.colors.ListedColormap
        Qualitative colormap.
    seq_cm : matplotlib.colors.ListedColormap
        Sequential colormap
    ensure_seq : bool
        Determines whether the sequential colormap
        is used independent of the number of colors.

    '''

    num_colors = abs(int(num_colors))

    # use qualitative colormap (if it has enough different colors)
    if (not ensure_seq) and (num_colors <= qual_cm.N):
        colors = [qual_cm(idx) for idx in range(num_colors)]

    # use sequential colormap (for an arbitrary number of colors)
    else:
        colors = seq_cm(np.linspace(0, 1, num_colors))

    return colors

