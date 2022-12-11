import click
from ...utils import OptionEatAll


codes = {
    "best": 0,
    "upper right": 1,  # default
    "upper left": 2,
    "lower left": 3,
    "lower right": 4,
    "center left": 5,
    "center right": 6,
    "lower center": 7,
    "upper center": 8,
    "center": 9,
    "top right": 10,
    "top left": 11,
    "bottom left": 12,
    "bottom right": 13,
    "right": 14,
    "left": 15,
    "top": 16,
    "bottom": 17,
}


@click.command()
@click.argument(
    "file_wannier90",
    type=click.Path(exists=True),
    required=False,
    default="wannier90_band.dat",
)
@click.argument("file_vasp", type=click.Path(exists=True), required=False, default=None)
@click.option("-s", "--spin", type=int, help="specify spin chanel")
@click.option(
    "-yr", "--yrange", type=(float, float), default=[-4, 6], help="energy range to plot"
)
@click.option(
    "-loc",
    "--location",
    type=(float, float),
    default=None,
    help="the location of lengend",
)
@click.option(
    "-cloc",
    "--codelocation",
    default=0,
    help=f"the location code of lengend, string of them are {codes}",
    type=click.IntRange(0, 17),
)
@click.option(
    "-vf",
    "--vaspfileformat",
    default="xml",
    required=False,
    envvar="MXMF_VASPFILE_FORMAT",
    type=click.Choice(["hdf5", "h5", "xml"]),
)
@click.option(
    "-xl",
    "--xlabel",
    cls=OptionEatAll,
    mytype="strlist",
    type=tuple,
    default=None,
    help="specify high symmetry kpoints label list ",
)
@click.option(
    "--save",
    cls=OptionEatAll,
    mytype="strlist",
    type=tuple,
    default=None,
    help="savefig as",
)
def compare(**kwargs):
    vaspfileformat = kwargs.pop("vaspfileformat")
    filew = kwargs.pop("file_wannier90")
    filev = kwargs.pop("file_vasp")
    savelist = kwargs.pop("save")
    import matplotlib.pyplot as plt
    from ...figplot import plot_wannier_band, plotband
    import matplotlib as mpl
    import mxmftools

    try:
        mpl.rc_file(f"~/.config/mxmf/matplotlibrc")
    except:
        mpl.rc_file(f"{mxmftools.__path__[0]}/matplotlibrc")

    if vaspfileformat == "hdf5" or vaspfileformat == "h5":
        from ...data_read import Readvaspout

        if filev is None:
            filev = "vaspout.h5"
        data = Readvaspout(filev)
    elif vaspfileformat == "xml":
        from ...data_read import ReadVasprun

        if filev is None:
            filev = "vasprun.xml"
        data = ReadVasprun(filev)
    fig, ax = plt.subplots()
    plotband(
        data,
        fig=fig,
        ax=ax,
        label=["vasp"],
        spin=kwargs["spin"],
        xlabel=kwargs["xlabel"],
    )
    plot_wannier_band(filew, fig=fig, ax=ax, fermi=data.fermi, yrange=kwargs["yrange"])
    if kwargs["location"] is None:
        ax.legend(loc=kwargs["codelocation"])
    else:
        ax.legend(loc=kwargs["location"])
    if savelist:
        for figname in savelist:
            plt.savefig(f"{figname}")
    else:
        plt.savefig(f"compare.png")
        plt.show()
