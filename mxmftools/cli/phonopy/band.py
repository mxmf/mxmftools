import click
from ...utils import OptionEatAll, add_options


@click.command
def band():
    from ...figplot.phononband import plot_band
    from ...data_read.read_kpoints import ReadKpoints
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import mxmftools

    try:
        mpl.rc_file(f"~/.config/mxmf/matplotlibrc")
    except:
        mpl.rc_file(f"{mxmftools.__path__[0]}/matplotlibrc")
    kpoints = ReadKpoints()

    fig, ax = plt.subplots()
    plot_band(kpoints.k_coors, kpoints.symbols, ax, kpoints.division)
    plt.savefig("phonon.png")
    plt.show()
