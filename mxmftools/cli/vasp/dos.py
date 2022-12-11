import click
from ...utils import OptionEatAll, add_options
from .common import vasp_common_options


@click.command
@click.argument("file", type=click.Path(exists=True), required=False)
@add_options(vasp_common_options)
@click.option("--fill", is_flag=True, default=False, help="plot using fill method")
@click.option("-g", "--gradient", is_flag=True, default=False, help=" gradient fill ")
@click.option("-p", "--pdos", is_flag=True, default=False, help="plot projected dos")
@click.option("-t", "--tdos", is_flag=True, default=False, help="plot total dos")
@click.option("-yr", "--yrange", type=(float, float), help="y range to plot")
@click.option("-xr", "--xrange", type=(float, float), help="x range to plot")
@click.option(
    "--alpha",
    type=tuple,
    cls=OptionEatAll,
    mytype="floatlist",
    help="specify transparency",
)
def dos(**kwargs):
    if kwargs["pdos"] is None and (kwargs["atoms"] or kwargs["orbitals"]):
        kwargs["pdos"] = True
    file = kwargs.pop("file")
    savelist = kwargs.pop("save")
    vaspfileformat = kwargs.pop("vaspfileformat")
    import matplotlib.pyplot as plt
    from ...figplot import plotdos
    from ...data_read import Readvaspout
    import matplotlib as mpl
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
    elif vaspfileformat == "xml":
        from ...data_read import ReadVasprun

        if file is None:
            file = "vasprun.xml"
        data = ReadVasprun(file)

    plotdos(data, **kwargs)
    if savelist:
        for figname in savelist:
            plt.savefig(f"{figname}")
    else:
        plt.savefig(f"dos.png")
        plt.show()
