# -*- coding: utf-8 -*-
# more colorstyles information see https://matplotlib.org/stable/tutorials/colors/colors.html
from matplotlib.ticker import AutoMinorLocator, AutoLocator
from ..data_read import Readvaspout
from ..data_handle import get_prolist
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
import colorcet as cc


def fillplot(x, y, color, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    ax.fill(x, y, color=color, **kwargs)


def lineplot(x, y, color, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    ax.plot(x, y, color=color, **kwargs)


# copy from https://stackoverflow.com/questions/29321835/is-it-possible-to-get-color-gradients-under-curve-in-matplotlib
def gradient_fill(x, y, ax=None, fill_color=None, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """
    if ax is None:
        ax = plt.gca()

    (line,) = ax.plot(x, y, **kwargs)
    if fill_color is None:
        fill_color = line.get_color()

    zorder = line.get_zorder()
    alpha = line.get_alpha()
    alpha = 1.0 if alpha is None else alpha

    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(fill_color)
    z[:, :, :3] = rgb
    z[:, :, -1] = np.linspace(0, alpha, 100)[:, None]
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    if ymin < 0:
        ymin, ymax = ymax, ymin
    im = ax.imshow(
        z, aspect="auto", extent=[xmin, xmax, ymin, ymax], origin="lower", zorder=zorder
    )

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = Polygon(xy, facecolor="none", edgecolor="none", closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    # ax.autoscale(True)
    return line, im


def single_plot(
    xlist,
    ylist,
    ax=None,
    spin=None,
    color=None,
    fill=False,
    alpha=None,
    gradient=False,
):
    if color is None:
        color = "red"
    if fill:
        if gradient:
            if alpha is None:
                alpha = 0.8
            plot = gradient_fill
        else:
            if alpha is None:
                alpha = 0.5
            plot = fillplot
    else:
        if alpha is None:
            alpha = 1
        plot = lineplot

    if ax is None:
        fig, ax = plt.subplots()
    ax.axvline(
        x=0,
        color="lightgrey",
        linestyle="--",
        linewidth=mpl.rcParams["ytick.major.width"],
        zorder=0,
    )
    ax.axhline(
        y=0,
        color="black",
        linestyle="-",
        linewidth=mpl.rcParams["ytick.major.width"],
        zorder=0,
    )

    if spin is None:
        if ylist.shape[0] == 2:
            # ax.set_ylim(-ylist[1].max(), ylist[0].max())
            plot(xlist, ylist[0], color=color, ax=ax, zorder=1, alpha=alpha)
            plot(xlist, -ylist[1], color=color, ax=ax, zorder=1, alpha=alpha)
        else:
            plot(xlist, ylist[0], color=color, ax=ax, zorder=1, alpha=alpha)
    else:
        plot(xlist, ylist[spin], color=color, ax=ax, zorder=1, alpha=alpha)


def plotdos(
    data,
    ax=None,
    fill=False,
    xrange=None,
    yrange=None,
    tdos=True,
    pdos=False,
    spin=None,
    color=None,
    alpha=None,
    gradient=False,
    atoms=None,
    orbitals=None,
):
    if ax is None:
        fig, ax = plt.subplots()
    if color is None:
        if pdos and tdos:
            colorlist = ["grey"] + cc.glasbey_category10
        else:
            colorlist = cc.glasbey_category10
    else:
        colorlist = color
    if alpha is None:
        alphalist = [None] * 20
    else:
        alphalist = list(alpha) + [None] * 20
    fermi = data.fermi
    dose = data.dose - fermi
    dos = data.dos
    if pdos:
        prolist = get_prolist(atoms, orbitals, data.symbols, data.dospar.shape[2])
        if tdos:
            single_plot(dose, dos, ax, spin, colorlist[0], fill, alphalist[0], gradient)
        for i, group in enumerate(prolist):
            if tdos:
                j = i + 1
            else:
                j = i
            ylist = sum(
                data.dospar[:, atoms, :, :][:, :, orbitals, :].sum(axis=1).sum(axis=1)
                for atoms, orbitals in group
            )
            single_plot(
                dose, ylist, ax, spin, colorlist[j], fill, alphalist[j], gradient
            )
    else:
        single_plot(dose, dos, ax, spin, colorlist[0], fill, alphalist[0], gradient)

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Density (states/eV)")
    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.autoscale(True)
    if xrange:
        ax.set_xlim(xrange)
    if yrange:
        ax.set_ylim(yrange)
