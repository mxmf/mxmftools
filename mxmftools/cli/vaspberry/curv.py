import click
from ...utils import OptionEatAll


@click.command
@click.argument(
    "file", type=click.Path(exists=True), default="BERRYCURV.dat", required=False
)
@click.option("-b", "--brillouin", type=click.Choice(["hex", "square"]))
@click.option("-i", "--interpolate", is_flag=True, default=False)
@click.option(
    "-m",
    "--intermethod",
    type=click.Choice(["nearest", "linear", "cubic"]),
    default="linear",
    help="interpolate method",
)
@click.option(
    "--save",
    type=tuple,
    cls=OptionEatAll,
    mytype="strlist",
    default=None,
    help="savefig as",
)
def curv(**kwargs):
    savelist = kwargs.pop("save")
    from ...figplot.vaspberry import plot_curvature
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import mxmftools

    try:
        mpl.rc_file(f"~/.config/mxmf/matplotlibrc")
    except:
        mpl.rc_file(f"{mxmftools.__path__[0]}/matplotlibrc")

    plot_curvature(**kwargs)
    if savelist:
        for figname in savelist:
            plt.savefig(f"{figname}")
    else:
        plt.savefig(f"curv.png")
        plt.show()
