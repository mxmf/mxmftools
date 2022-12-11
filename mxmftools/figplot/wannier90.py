import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('/home/chenzp/wuxc/pythonscript/matplotlibrc')


def plot_wannier_band(
    file="wannier90_band.dat",
    yrange=[-6, 4],
    fig=None,
    ax=None,
    fermi=0,
    color="black",
    size=1,
):
    xlist = []
    ylist = [[]]
    with open(file, "r") as fw:
        bandnum = 0
        for linenum, line in enumerate(fw.readlines()):
            if line.isspace():
                ylist.append([])
                bandnum += 1
            else:
                if bandnum == 0:
                    xlist.append(float(line.split()[0]))
                ylist[bandnum].append(float(line.split()[1]))
    ylist.pop(-1)
    ylist = np.array(ylist).transpose() - fermi
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_ylim(yrange)
    ax.set_xlim(xlist[0], xlist[-1])
    lines = ax.plot(xlist, ylist, color="black", ls=":")
    plt.setp(lines[0], label="wannier")
