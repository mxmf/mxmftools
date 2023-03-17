from ..cli.vasp.band import band
from ..figplot import plotband


def plotband_from_cli_str(data, str, fig=None, ax=None):
    import matplotlib as mpl
    import mxmftools

    try:
        mpl.rc_file(f"~/.config/mxmf/matplotlibrc")
    except:
        mpl.rc_file(f"{mxmftools.__path__[0]}/matplotlibrc")
    kwargs = band.make_context("test", str.split()).params
    kwargs.pop("file")
    kwargs.pop("save")
    kwargs.pop("vaspfileformat")
    plotband(data=data, **kwargs, fig=fig, ax=ax)
