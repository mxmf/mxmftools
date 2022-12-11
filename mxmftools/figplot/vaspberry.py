import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import colorcet as cc
from matplotlib.path import Path
import matplotlib.patches as patches


def read_comments(file="BERRYCURV.dat"):
    comments_dic = {}
    with open(file) as fvb:
        for line in fvb:
            if line[0] == "#" or line.isspace():
                line = line.replace("#", "")
                if "ISPIN" in line:
                    txt = re.split(":|\(", line.replace(" ", ""))
                    comments_dic.update({"ISPIN": int(txt[1])})
                elif "K-GRID" in line:
                    txt = line.split(":")
                    value = np.array([int(i) for i in txt[1].split("X")])
                    comments_dic.update({"K-GRID": value})
                elif "RECIVEC" in line:
                    txt = line.split(":")
                    key = txt[0].split()[1]
                    value = np.array([float(i) for i in txt[1].split()])
                    comments_dic.update({key: value})
                elif "Chern Number for the BANDS" in line:
                    value = [int(i) for i in line.split(":")[1].split("-")]
                    comments_dic.update({"Chern Number for the BANDS": value})
                elif "of BERRYCURV at kx,ky,kz" in line:
                    txt = line.split("=")
                    key = txt[0].split()[0]
                    value = np.array([float(i) for i in txt[1].split()])
                    comments_dic.update({key: value})
                elif "dk^2 = |dk1xk2|" in line:
                    value = float(line.split()[-1])
                    comments_dic.update({"dk^2": value})
                else:
                    txt = line.split(":")
                    if len(txt) == 2:
                        key = txt[0].strip()
                        try:
                            comments_dic.update({key: int(txt[1])})
                        except ValueError:
                            try:
                                comments_dic.update({key: float(txt[1])})
                            except:
                                comments_dic.update({key: txt[1].strip()})
                    txt = line.split("=")
                    if len(txt) == 2:
                        key = txt[0].strip()
                        try:
                            comments_dic.update({key: int(txt[1])})
                        except ValueError:
                            try:
                                comments_dic.update({key: float(txt[1])})
                            except:
                                comments_dic.update({key: txt[1].strip()})
            else:
                break
    return comments_dic


def plot_curvature(
    file="BERRYCURV.dat", interpolate=False, intermethod="linear", brillouin=None
):
    kx, ky, kz, Berry_Curvature, kx_r, ky_r, kz_r = np.loadtxt(file, unpack=True)
    comments = read_comments(file)
    grid = comments["K-GRID"] * 3
    # # no interpolate
    if interpolate:
        from scipy.interpolate import griddata

        kx_inter = np.linspace(kx.min(), kx.max(), 400)
        ky_inter = np.linspace(ky.min(), ky.max(), 400)
        Berry_Curvature_mesh = griddata(
            (kx, ky),
            Berry_Curvature,
            (kx_inter[None, :], ky_inter[:, None]),
            method=intermethod,
        )
        kx_mesh, ky_mesh = np.meshgrid(kx_inter, ky_inter)
    else:
        kx_mesh, ky_mesh, Berry_Curvature_mesh = (
            kx.reshape(grid),
            ky.reshape(grid),
            Berry_Curvature.reshape(grid),
        )

    fig, ax = plt.subplots(figsize=[3, 3])
    ax.set_aspect("equal", "box")
    ax.set_axis_off()
    norm = mpl.colors.Normalize(Berry_Curvature.min(), Berry_Curvature.max())
    img = ax.pcolormesh(
        kx_mesh,
        ky_mesh,
        Berry_Curvature_mesh,
        cmap=cc.m_coolwarm,
        norm=norm,
    )
    cbar = fig.colorbar(
        img, ax=ax, ticks=np.linspace(Berry_Curvature.min(), Berry_Curvature.max(), 5)
    )
    if brillouin is not None:
        b1, b2 = comments["B1"][:2], comments["B2"][:2]
        if brillouin == "hex":
            high_K = [
                (1 / 3, 1 / 3),
                (2 / 3, -1 / 3),
                (1 / 3, -2 / 3),
                (-1 / 3, -1 / 3),
                (-2 / 3, 1 / 3),
                (-1 / 3, 2 / 3),
            ]
        elif brillouin == "square":
            high_K = [
                (1 / 2, 1 / 2),
                (1 / 2, -1 / 2),
                (-1 / 2, -1 / 2),
                (-1 / 2, 1 / 2),
            ]
        else:
            print("only surport for hex and square, if you need other, tell me")
            import sys

            sys.exit()
        verts = [np.dot(K, (b1, b2)) for K in high_K] + [np.dot(high_K[0], (b1, b2))]
        codes = [Path.MOVETO] + (len(verts) - 2) * [Path.LINETO] + [Path.CLOSEPOLY]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor="none", lw=2)
        ax.add_patch(patch)
