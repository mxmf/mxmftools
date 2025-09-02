from .params import CompareParams
from mxmftools.vasp import BandPlot
from matplotlib import figure, axes
import numpy as np


class ComparePlot(BandPlot):
    def __init__(self, params: CompareParams, fig: figure.Figure, ax: axes.Axes):
        super().__init__(params, fig, ax)
        self._compare_params = params
        self.plot_wannier_band()

    def plot_wannier_band(self):
        k, e = np.loadtxt(self._compare_params.wannierfile, unpack=True)
        knum = 0
        with open(self._compare_params.wannierfile) as fw:
            for i, line in enumerate(fw):
                if line.isspace():
                    knum = i
                    break

        e = e - self.fermi
        ks = k.reshape(-1, knum).transpose()
        es = e.reshape(-1, knum).transpose()
        self.ax.plot(ks, es, ls=":", c="black", label="wannier fit")
