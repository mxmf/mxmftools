import phonopy
import numpy as np
import matplotlib.pyplot as plt
from phonopy.phonon.band_structure import get_band_qpoints
import matplotlib as mpl


def plot_band(kcoor, labels, ax, division=51):
    phonon = phonopy.load("phonopy_disp.yaml")
    points = get_band_qpoints([kcoor], division)
    phonon.run_band_structure(points, labels=labels)
    if ax is None:
        fig, ax = plt.subplots()
    bs = phonon._band_structure
    a = np.array(bs.frequencies)
    b = a.reshape(-1, a.shape[-1])
    ax.plot(b, color="blue", zorder=2)
    ax.set_xmargin(0)
    ax.axhline(
        0, ls="--", color="black", zorder=1, lw=mpl.rcParams["ytick.major.width"]
    )

    xticks = [i * division for i in range(4)]
    for xtick in xticks[1:-1]:
        ax.axvline(
            xtick, color="lightgrey", zorder=1, lw=mpl.rcParams["xtick.major.width"]
        )
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Frequency (THz)")
    ax.tick_params(bottom=False)
