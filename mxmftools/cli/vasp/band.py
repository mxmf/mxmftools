import click
from ...utils import OptionEatAll, add_options
from .common import vasp_common_options


@click.command
@click.argument("file", type=click.Path(exists=True), default=None, required=False)
@add_options(vasp_common_options)
@click.option("-f", "--fermi", type=float, help="fermi engergy")
@click.option("-p", "--pband", type=click.IntRange(0, 4), help="specify projected mode")
@click.option("-d", "--division", type=float, help="division of neibor kpoints")
@click.option("-cb/-nocb", default=True, help="whether plot colorbar")
@click.option("-hse/-nohse", default=False, help="whether plot zero weight kpoints")
@click.option("-xr", "--xrange", type=(int, int), help="K range to plot")
@click.option(
    "-xt",
    "--xtickindex",
    type=tuple,
    cls=OptionEatAll,
    mytype="strlist",
    help="specify high symmetry kpoints index list",
)
@click.option(
    "-xl",
    "--xlabel",
    type=tuple,
    cls=OptionEatAll,
    mytype="strlist",
    help="specify high symmetry kpoints label list ",
)
def band(**kwargs):
    if kwargs["pband"] is None and (kwargs["atoms"] or kwargs["orbitals"]):
        kwargs["pband"] = 0
    file = kwargs.pop("file")
    savelist = kwargs.pop("save")
    vaspfileformat = kwargs.pop("vaspfileformat")
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from ...figplot import plotband
    import mxmftools

    try:
        mpl.rc_file(f"~/.config/mxmf/matplotlibrc")
    except:
        mpl.rc_file(f"{mxmftools.__path__[0]}/matplotlibrc")

    if vaspfileformat == "hdf5" or vaspfileformat == "h5":
        from ...data_read import Readvaspout

        if file is None:
            file = "vaspout.h5"
        data = Readvaspout(file)
    else:
        from ...data_read import ReadVasprun

        if file is None:
            file = "vasprun.xml"
        data = ReadVasprun(file)

    plotband(data, **kwargs)
    if savelist:
        for figname in savelist:
            plt.savefig(f"{figname}")
    else:
        plt.savefig(f"band.png")
        plt.show()
