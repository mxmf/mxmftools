import click


@click.command()
@click.argument(
    "file",
    type=click.Path(exists=True),
    required=False,
    default="wannier90_band.dat",
)
@click.option(
    "-yr", "--yrange", type=(float, float), default=[-4, 6], help="energy range to plot"
)
@click.option("-f", "--fermi", default=0, help="fermi engergy", type=float)
def band(**kwargs):
    import matplotlib.pyplot as plt
    from ...figplot import plot_wannier_band
    import matplotlib as mpl
    import mxmftools

    try:
        mpl.rc_file(f"~/.config/mxmf/matplotlibrc")
    except:
        mpl.rc_file(f"{mxmftools.__path__[0]}/matplotlibrc")

    file = kwargs["file"]
    plot_wannier_band(file, fermi=kwargs["fermi"], yrange=kwargs["yrange"])
    plt.show()
