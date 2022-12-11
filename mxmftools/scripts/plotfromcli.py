from ..cli import band
from ..figplot import plotband


def plotband_from_cli_str(data, str, fig=None, ax=None):
    kwargs = band.make_context("test", str.split()).params
    kwargs.pop("file")
    kwargs.pop("save")
    kwargs.pop("readfileformat")
    plotband(data=data, **kwargs, fig=fig, ax=ax)
