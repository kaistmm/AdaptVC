import matplotlib

matplotlib.use("Agg")
import matplotlib.pylab as plt


def plot_specs(specs, titles=None):
    if titles is None:
        titles = [None] * len(specs)
    fig, ax = plt.subplots(len(specs), 1, figsize=(10, 2 * len(specs)))
    for idx in range(len(specs)):
        im = ax[idx].imshow(
            specs[idx], aspect="auto", origin="lower", interpolation="none"
        )
        # remove axis
        ax[idx].set_xticks([])
        plt.colorbar(im, ax=ax[idx])
        ax[idx].set_title(titles[idx])
    fig.canvas.draw()
    plt.close()

    return fig


def plot_weight(weight):
    fig, ax = plt.subplots(2, 1, figsize=(10, 4))
    im = ax[0].imshow(weight, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    im = ax[1].plot(weight[0])
    fig.canvas.draw()
    plt.close()

    return fig


import sys
import pdb


class ForkedPdb(pdb.Pdb):
    """
    PDB Subclass for debugging multi-processed code
    Suggested in: https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
