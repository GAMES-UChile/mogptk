import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
warnings.simplefilter('ignore', UserWarning)

def plot_gram(K):
    fig, ax = plt.subplots(1, 1, figsize=(12,12))
    fig.suptitle('Matrix is not positive semi-definitive', fontsize=20)

    K = K.detach().numpy()
    #K_real = np.ma.masked_where(np.isnan(K)|np.isinf(K), K)
    K_real = K[~np.isnan(K) & ~np.isinf(K)]
    norm = matplotlib.colors.Normalize(vmin=np.abs(K_real).min(), vmax=np.abs(K_real).max())
    im = ax.matshow(np.where(np.isinf(K)|np.isnan(K),np.nan,K), cmap='viridis', norm=norm)

    # show Inf and NaN as blue and red respectively
    cmap = matplotlib.colors.ListedColormap(["blue"])
    ax.matshow(np.where(np.isinf(K),1.0,np.nan), cmap=cmap)
    cmap = matplotlib.colors.ListedColormap(["red"])
    ax.matshow(np.where(np.isnan(K),1.0,np.nan), cmap=cmap)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    fig.colorbar(im, cax=cax)

    ax.set_title('Blue=Inf, Red=NaN', pad=30)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()
