# -*- coding: utf-8 -*-
import numpy as np
import colorcet as cc
import matplotlib as mpl
import itertools
from ..data_handle import get_prolist, get_gap
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import AutoMinorLocator, AutoLocator


def get_xticks(xlist, xtickindex, xlabel, division):
    if xtickindex:
        xticks = [xlist[i] for i in xtickindex]
    elif division:
        xticks = ([0] + xlist)[0::division]
    else:
        xticks = [0]
        for i, x in enumerate(xlist):
            if i > 0 and x - xlist[i - 1] < 0.0000001:
                xticks.append(x)
        xticks.append(xlist[-1])
    if xlabel:
        xticklabels = xlabel
    else:
        print(
            "There is no label read from kpoints, use fake labels to plot fig, check it!!! "
        )
        xticklabels = [
            "M",
            "K",
            "$\Gamma$",
            "K",
            "$\Gamma$",
            "$\Gamma$",
            "$\Gamma$",
            "$\Gamma$",
            "$\Gamma$",
        ][0 : len(xticks)]
    return xticks, xticklabels


def get_proarray(projected, prolist):
    linearray = []
    pointsarray = []
    for group in prolist:
        points = sum(
            projected[:, atoms, :, :, :][:, :, orbitals, :, :].sum(axis=1).sum(axis=1)
            for atoms, orbitals in group
        )
        pointsarray.append(points)
        line = points[:, :-1, :]
        linearray.append(line.transpose(0, 2, 1).reshape(len(projected), -1))
    return np.array(linearray), np.array(pointsarray)


def kpointstoxlist(kpoints, rec_cell):
    kpoints_real = [np.dot(kpoint, rec_cell) for kpoint in kpoints]

    length = (
        np.linalg.norm(kpoint1 - kpoint2)
        for kpoint1, kpoint2 in zip(
            kpoints_real, kpoints_real[0:1] + list(kpoints_real)[:-1]
        )
    )
    xlist = list(itertools.accumulate(length))
    return xlist


def plot_band(xlist, ylist, ax, xrange, yrange, xticks, colorlist, spin, label):
    if colorlist is None:
        colorlist = ["red", "blue"]
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_ylabel("Energy(eV)")
    ax.set_ylim(yrange)
    ax.set_xlim(xlist[0], xlist[-1])
    ax.axhline(
        y=0,
        linestyle="--",
        color="black",
        linewidth=mpl.rcParams["ytick.major.width"],
        zorder=0,
    )
    for xtick in xticks[0]:
        ax.axvline(
            x=xtick,
            linestyle="-",
            linewidth=mpl.rcParams["xtick.major.width"],
            color="lightgrey",
            zorder=0,
        )
    ax.set_xticks(xticks[0])
    ax.set_xticklabels(xticks[1])
    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(bottom=False)
    if xrange:
        ax.set_xlim(xlist[xrange[0]], xlist[xrange[1]])
    # write_band_dat(xlist, ylist[0], 'band_up.dat')
    # if len(ylist) == 2:
    # write_band_dat(xlist, ylist[1], 'band_down.dat')
    if spin is None:
        ups = ax.plot(xlist, ylist[0], color=colorlist[0], zorder=1)
        plt.setp(ups[0], label=label[0])
        if len(ylist) == 2:
            dns = ax.plot(xlist, ylist[1], color=colorlist[1], zorder=1)
            plt.setp(dns[0], label=label[1])

    else:
        ups = ax.plot(xlist, ylist[spin], color=colorlist[0], zorder=1)
        plt.setp(ups[0], label=label[0])


def proplot1(xlist, ylist, plotarray, cmap, ax, fig, colorbar):
    xlist1 = np.tile(np.array(xlist).repeat(2)[1:-1], ylist.shape[1])
    ylist1 = ylist[:, :].repeat(2, axis=0)[1:-1, :].reshape(-1, order="F")
    norm = mpl.colors.Normalize(0, plotarray.max())
    lc = LineCollection(
        np.array(list(zip(xlist1, ylist1))).reshape(-1, 2, 2), cmap=cmap, norm=norm
    )
    lc.set_capstyle("round")
    lc.set_array(plotarray)
    ax.add_collection(lc)
    if colorbar:
        cbar = fig.colorbar(lc, ax=ax, ticks=np.linspace(0, np.max(plotarray), 5))
        cbar.ax.set_yticklabels(np.around(np.linspace(0, np.max(plotarray), 5), 2))


def proplot2(xlist, ylist, plotarray, cmap, ax, fig, colorbar):
    xlist1 = np.tile(np.array(xlist).repeat(2)[1:-1], ylist.shape[1])
    ylist1 = ylist[:, :].repeat(2, axis=0)[1:-1, :].reshape(-1, order="F")
    norm = mpl.colors.TwoSlopeNorm(
        vmin=plotarray.min(), vmax=plotarray.max(), vcenter=0
    )
    lc = LineCollection(
        np.array(list(zip(xlist1, ylist1))).reshape(-1, 2, 2), cmap=cmap, norm=norm
    )
    lc.set_capstyle("round")
    lc.set_array(plotarray)
    ax.add_collection(lc)
    if colorbar:
        cbar = fig.colorbar(
            lc,
            ax=ax,
            ticks=[
                np.min(plotarray),
                np.min(plotarray) / 2,
                0,
                np.max(plotarray) / 2,
                np.max(plotarray),
            ],
        )
        cbar.ax.set_yticklabels(
            np.around(
                [
                    np.min(plotarray),
                    np.min(plotarray) / 2,
                    0,
                    np.max(plotarray) / 2,
                    np.max(plotarray),
                ],
                2,
            )
        )


def proplot3(xlist, ylist, pointsarray, ax, color, zorder):
    pointsarray[pointsarray < 0.01] = 0.01
    ax.scatter(xlist, ylist, s=pointsarray, c=color, marker="o", zorder=zorder)


# # def proplot5(plotarray, xlist, ylist, args, ax):
# #     xlist1 = np.tile(np.array(xlist).repeat(2)[1:-1], ylist.shape[1])
# #     ylist1 = ylist[:, :].repeat(2, axis=0)[1:-1, :].reshape(-1, order="F")
# #     lc = LineCollection(np.array(list(zip(xlist1, ylist1))
# #                                  ).reshape(-1, 2, 2), color=plotarray)
# #     lc.set_capstyle('round')
# #     ax.add_collection(lc)


def plot_proband(
    xlist,
    ylist,
    protuple,
    ax,
    fig,
    xticks,
    xrange,
    yrange,
    pband,
    spin,
    color,
    colorbar,
):
    linearray, plotarray = protuple
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_ylabel("Energy(eV)")
    ax.set_ylim(yrange)
    ax.set_xlim(xlist[0], xlist[-1])
    ax.axhline(
        y=0,
        linestyle="--",
        color="black",
        linewidth=mpl.rcParams["ytick.major.width"],
        zorder=0,
    )
    for xtick in xticks[0]:
        ax.axvline(
            x=xtick,
            linestyle="-",
            linewidth=mpl.rcParams["xtick.major.width"],
            color="lightgrey",
            zorder=0,
        )
    ax.set_xticks(xticks[0])
    ax.set_xticklabels(xticks[1])
    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(bottom=False)
    if xrange:
        ax.set_xlim(xlist[xrange[0]], xlist[xrange[1]])

    if pband == 0:
        if color is None:
            cmap1 = cc.m_CET_L18
            cmap2 = cc.m_coolwarm
        else:
            cmap1 = color[0]
            cmap2 = color[0]
        # Line Collection with mapped colors
        plotarray = linearray.sum(axis=0)
        if spin is None:
            if len(ylist) == 2:
                ylist = np.concatenate((ylist[0], ylist[1]), axis=1)
                plotarray = np.concatenate((plotarray[0], -plotarray[1]), axis=0)
                proplot2(xlist, ylist, plotarray, cmap2, ax, fig, colorbar)
            else:
                proplot1(xlist, ylist[0], plotarray[0], cmap1, ax, fig, colorbar)
        elif spin == 0:
            proplot1(xlist, ylist[0], plotarray[0], cmap1, ax, fig)
        else:
            if len(plotarray) == 4:
                proplot2(xlist, ylist[0], plotarray[spin], cmap2, ax, fig, colorbar)
            else:
                proplot1(xlist, ylist[spin], plotarray[spin], cmap1, ax, fig, colorbar)

    elif pband == 1:
        if color:
            color_list = color
            ...
        else:
            color_list = [
                "red",
                "blue",
                "yellow",
                "lime",
            ]
        pointsarray = plotarray.sum(axis=0)
        xlist1 = np.tile(np.array(xlist).reshape(-1, 1), ylist.shape[-1])
        if spin is None:
            scale = mpl.rcParams["lines.markersize"] / np.max(pointsarray)
            if len(ylist) == 2:
                proplot3(xlist1, ylist[0], pointsarray[0] * scale, ax, color_list[0], 1)
                proplot3(xlist1, ylist[1], pointsarray[1] * scale, ax, color_list[1], 1)
            else:
                proplot3(xlist1, ylist[0], pointsarray[0] * scale, ax, color_list[0], 1)
        elif spin == 0:
            scale = mpl.rcParams["lines.markersize"] / np.max(pointsarray[0])
            proplot3(xlist1, ylist[0], pointsarray[0] * scale, ax, color_list[0], 1)
        else:
            if len(pointsarray) == 4:
                pointsarray = pointsarray[spin]
                scale = mpl.rcParams["lines.markersize"] / np.max(np.abs(pointsarray))
                up_index = np.where(pointsarray > 0)
                down_index = np.where(pointsarray < 0)
                proplot3(
                    xlist1[up_index],
                    ylist[0][up_index],
                    pointsarray[up_index] * scale,
                    ax,
                    color_list[0],
                    1,
                )
                proplot3(
                    xlist1[down_index],
                    ylist[0][down_index],
                    -pointsarray[down_index] * scale,
                    ax,
                    color_list[1],
                    1,
                )
            else:
                scale = mpl.rcParams["lines.markersize"] / np.max(pointsarray[spin])
                proplot3(
                    xlist1, ylist[spin], pointsarray[spin] * scale, ax, color_list[0], 1
                )

    elif pband == 2:
        if color:
            color_list = color
        else:
            color_list = cc.glasbey_light
            color_list = ["blue", "red", "green", "orange"]
        pointsarray = plotarray.transpose(1, 0, 2, 3).reshape(plotarray.shape[1], -1)
        c = np.array(color_list[: plotarray.shape[0]]).repeat(
            ylist.shape[1] * ylist.shape[2]
        )
        xlist1 = np.tile(np.array(xlist).repeat(ylist.shape[2]), plotarray.shape[0])
        ylist1 = np.tile(ylist.reshape(ylist.shape[0], -1), plotarray.shape[0])
        if spin is None:
            if len(ylist) == 2:
                scale = mpl.rcParams["lines.markersize"] / np.max(pointsarray)
                ax.scatter(xlist1, ylist1[0], s=pointsarray[0] * scale, c=c, zorder=1)
                ax.scatter(xlist1, ylist1[1], s=pointsarray[1] * scale, c=c, zorder=1)
            else:
                scale = mpl.rcParams["lines.markersize"] / np.max(pointsarray[0])
                ax.scatter(xlist1, ylist1[0], s=pointsarray[0] * scale, c=c, zorder=1)
        elif spin == 0:
            scale = mpl.rcParams["lines.markersize"] / np.max(pointsarray[0])
            ax.scatter(xlist1, ylist1[0], s=pointsarray[0] * scale, c=c, zorder=1)
        else:
            if len(pointsarray) == 4:
                pointsarray = pointsarray[spin]
                scale = mpl.rcParams["lines.markersize"] / np.max(np.abs(pointsarray))
                up_index = np.where(pointsarray > 0)
                down_index = np.where(pointsarray < 0)
                ax.scatter(
                    xlist1[up_index],
                    ylist1[0][up_index],
                    pointsarray[up_index] * scale,
                    c[up_index],
                    zorder=1,
                )
                ax.scatter(
                    xlist1[down_index],
                    ylist1[0][down_index],
                    -pointsarray[down_index] * scale,
                    c[down_index],
                    zorder=1,
                )

            else:
                scale = mpl.rcParams["lines.markersize"] / np.max(pointsarray[spin])
                ax.scatter(
                    xlist1, ylist1[spin], pointsarray[spin] * scale, c=c, zorder=1
                )

    elif pband == 3:
        if color:
            color_list = color
        else:
            color_list = cc.glasbey_light
            color_list = ["blue", "red", "green", "orange"]
        pointsarray = plotarray.transpose(1, 0, 2, 3).reshape(plotarray.shape[1], -1)
        order = pointsarray.argsort(axis=1)[:, ::-1]
        pointsarray.sort(axis=1)
        pointsarray = pointsarray[:, ::-1]
        c = np.array(color_list[: plotarray.shape[0]]).repeat(
            ylist.shape[1] * ylist.shape[2]
        )
        xlist1 = np.tile(np.array(xlist).repeat(ylist.shape[2]), plotarray.shape[0])
        ylist1 = np.tile(ylist.reshape(ylist.shape[0], -1), plotarray.shape[0])
        if spin is None:
            if len(ylist) == 2:
                scale = mpl.rcParams["lines.markersize"] / np.max(pointsarray)
                ax.scatter(
                    xlist1[order[0]],
                    ylist1[0][order[0]],
                    s=pointsarray[0] * scale,
                    c=c[order[0]],
                    zorder=1,
                )
                ax.scatter(
                    xlist1[order[1]],
                    ylist1[1][order[1]],
                    s=pointsarray[1] * scale,
                    c=c[order[1]],
                    zorder=1,
                )
            else:
                scale = mpl.rcParams["lines.markersize"] / np.max(pointsarray[0])
                ax.scatter(
                    xlist1[order[0]],
                    ylist1[0][order[0]],
                    s=pointsarray[0] * scale,
                    c=c[order[0]],
                    zorder=1,
                )
        elif spin == 0:
            scale = mpl.rcParams["lines.markersize"] / np.max(pointsarray[0])
            ax.scatter(
                xlist1[order[0]],
                ylist1[0][order[0]],
                s=pointsarray[0] * scale,
                c=c[order[0]],
                zorder=1,
            )
        else:
            if len(pointsarray) == 4:
                pointsarray = pointsarray[spin]
                # print(pointsarray)
                scale = mpl.rcParams["lines.markersize"] / np.max(np.abs(pointsarray))
                up_index = np.where(pointsarray > 0)
                down_index = np.where(pointsarray < 0)
                x_up, x_down = (
                    xlist1[order[spin]][up_index],
                    xlist1[order[spin]][down_index],
                )
                y_up, y_down = (
                    ylist1[0][order[spin]][up_index],
                    ylist1[0][order[spin]][down_index],
                )
                c_up, c_down = c[order[spin]][up_index], c[order[spin]][down_index]

                ax.scatter(x_up, y_up, pointsarray[up_index] * scale, c_up, zorder=1)
                ax.scatter(
                    x_down, y_down, pointsarray[up_index] * scale, c_down, zorder=1
                )

            else:
                scale = mpl.rcParams["lines.markersize"] / np.max(pointsarray[spin])
                ax.scatter(
                    xlist1[order[spin]],
                    ylist1[spin][order[spin]],
                    pointsarray[spin] * scale,
                    c=c[order[spin]],
                    zorder=1,
                )

    elif pband == 4:
        if color:
            color_list = color
        else:
            color_list = cc.glasbey_light
            color_list = ["blue", "red", "green", "orange"]
        pointsarray = plotarray.transpose(1, 0, 2, 3).reshape(plotarray.shape[1], -1)
        order = pointsarray.argsort(axis=1)
        pointsarray.sort(axis=1)
        pointsarray = pointsarray
        c = np.array(color_list[: plotarray.shape[0]]).repeat(
            ylist.shape[1] * ylist.shape[2]
        )
        xlist1 = np.tile(np.array(xlist).repeat(ylist.shape[2]), plotarray.shape[0])
        ylist1 = np.tile(ylist.reshape(ylist.shape[0], -1), plotarray.shape[0])
        if spin is None:
            if len(ylist) == 2:
                scale = mpl.rcParams["lines.markersize"] / np.max(pointsarray)
                ax.scatter(
                    xlist1[order[0]],
                    ylist1[0][order[0]],
                    s=pointsarray[0] * scale,
                    c=c[order[0]],
                    zorder=1,
                )
                ax.scatter(
                    xlist1[order[1]],
                    ylist1[1][order[1]],
                    s=pointsarray[1] * scale,
                    c=c[order[1]],
                    zorder=1,
                )
            else:
                scale = mpl.rcParams["lines.markersize"] / np.max(pointsarray[0])
                ax.scatter(
                    xlist1[order[0]],
                    ylist1[0][order[0]],
                    s=pointsarray[0] * scale,
                    c=c[order[0]],
                    zorder=1,
                )
        elif spin == 0:
            scale = mpl.rcParams["lines.markersize"] / np.max(pointsarray[0])
            ax.scatter(
                xlist1[order[0]],
                ylist1[0][order[0]],
                s=pointsarray[0] * scale,
                c=c[order[0]],
                zorder=1,
            )
        else:
            if len(pointsarray) == 4:
                pointsarray = pointsarray[spin]
                # print(pointsarray)
                scale = mpl.rcParams["lines.markersize"] / np.max(np.abs(pointsarray))
                up_index = np.where(pointsarray > 0)
                down_index = np.where(pointsarray < 0)
                x_up, x_down = (
                    xlist1[order[spin]][up_index],
                    xlist1[order[spin]][down_index],
                )
                y_up, y_down = (
                    ylist1[0][order[spin]][up_index],
                    ylist1[0][order[spin]][down_index],
                )
                c_up, c_down = c[order[spin]][up_index], c[order[spin]][down_index]

                ax.scatter(x_up, y_up, pointsarray[up_index] * scale, c_up, zorder=1)
                ax.scatter(
                    x_down, y_down, pointsarray[up_index] * scale, c_down, zorder=1
                )

            else:
                scale = mpl.rcParams["lines.markersize"] / np.max(pointsarray[spin])
                ax.scatter(
                    xlist1[order[spin]],
                    ylist1[spin][order[spin]],
                    pointsarray[spin] * scale,
                    c=c[order[spin]],
                    zorder=1,
                )


#         # elif pband == 6:
#         #     plotarray = linearray.transpose(1, 2, 0)
#         #     if spin is None:
#         #         if len(ylist) == 2:
#         #             ylist = np.concatenate((ylist[0], ylist[1]), axis=1)
#         #             plotarray = np.concatenate(
#         #                 (plotarray[0], plotarray[1]), axis=0)
#         #             proplot5(xlist=xlist, ylist=ylist,
#         #                      plotarray=plotarray, args=args, ax=ax)
#         #         else:
#         #             proplot5(
#         #                 xlist=xlist, ylist=ylist[0], plotarray=plotarray[0] / plotarray[0].max(axis=0), args=args, ax=ax)
#         #     else:
#         #         proplot5(xlist=xlist, ylist=ylist[spin],
#         #                  plotarray=plotarray[spin], args=args, ax=ax)


def plotband(
    data,
    xtickindex=None,
    xlabel=None,
    division=None,
    fermi=None,
    pband=None,
    xrange=None,
    yrange=None,
    color=None,
    spin=None,
    atoms=None,
    orbitals=None,
    cb=None,
    hse=None,
    fig=None,
    ax=None,
    label=["spin up", "spin down"],
):
    weights = data.weights_opt
    if weights is None:
        weights = data.weights
        if hse:
            use_index = np.where(weights == 0)[0]
        else:
            use_index = np.where(weights >= 0)[0]

        kpoints = data.kpoints[use_index] * 2 * np.pi
        eigenvalues = data.eigenvalues[:, use_index, :]
        projected = data.projected[:, :, :, use_index, :]

        rec_cell = data.rec_cell
    else:
        if hse:
            use_index = np.where(weights == 0)[0]
        else:
            use_index = np.where(weights >= 0)[0]

        kpoints = data.kpoints_opt[use_index] * 2 * np.pi
        eigenvalues = data.eigenvalues_opt[:, use_index, :]
        projected = data.projected_opt[:, :, :, use_index, :]

        rec_cell = data.rec_cell
    if fermi is None:
        fermi = data.fermi
    else:
        fermi = float(fermi)
    if xlabel:
        xticklabels = xlabel
    else:
        try:
            xticklabels = data.labels_kpoints
        except AttributeError:
            xticklabels = None
    xlist = kpointstoxlist(kpoints, rec_cell)
    ylist = eigenvalues - fermi
    get_gap(ylist, kpoints, stdout=True)
    if ax is None:
        fig, ax = plt.subplots()
    xticks = get_xticks(xlist, xtickindex, xticklabels, division)
    if pband is None:
        plot_band(xlist, ylist, ax, xrange, yrange, xticks, color, spin, label)
    else:
        prolist = get_prolist(atoms, orbitals, data.symbols, projected.shape[2])
        protuple = get_proarray(projected, prolist)
        plot_proband(
            xlist,
            ylist,
            protuple,
            ax,
            fig,
            xticks,
            xrange,
            yrange,
            pband,
            spin,
            color,
            colorbar,
        )
