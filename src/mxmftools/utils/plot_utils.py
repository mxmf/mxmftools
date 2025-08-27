from typing import TYPE_CHECKING, cast

from matplotlib import colors
from typing_extensions import override

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib import figure

    # from ..cli import utils as cliutils
    from .common_params import FigSetBase, HeatFigBase


class MyCustomNormalize(colors.Normalize):
    """
    Modified from https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
    """

    def __init__(
        self, vmin: float, vmax: float, midpoint: float = 0.0, clip: bool = False
    ):
        self.midpoint: float = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    @override
    def __call__(self, value, clip: bool | None = None):
        vmin = cast(float, self.vmin)
        vmax = cast(float, self.vmax)
        import numpy as np

        if vmin == vmax:
            return np.full_like(value, 0.5, dtype=np.float64)
        midpoint = self.midpoint

        normalized_min = (
            max(
                0,
                1 / 2 * (1 - abs((midpoint - vmin) / (midpoint - vmax))),
            )
            if (self.midpoint != self.vmax)
            else 0
        )
        normalized_max = (
            min(
                1,
                1 / 2 * (1 + abs((vmax - midpoint) / (midpoint - vmin))),
            )
            if (self.midpoint != self.vmin)
            else 1.0
        )
        normalized_mid = 0.5
        x, y = (
            [self.vmin, self.midpoint, self.vmax],
            [normalized_min, normalized_mid, normalized_max],
        )
        return np.ma.masked_array(np.interp(value, x, y))


class AxesSet:
    def __init__(self, ax: "plt.Axes", params: "FigSetBase"):
        self.ax: plt.Axes = ax
        self.params: "FigSetBase" = params
        self.set_title()
        self.set_labels()
        self.set_xticks()
        self.set_yticks()
        self.set_lims()
        self.hide_ticks()

    def set_title(self):
        self.ax.set_title(self.params.title) if self.params.title is not None else ...

    def set_labels(self):
        self.ax.set_xlabel(
            self.params.xlabel
        ) if self.params.xlabel is not None else ...
        self.ax.set_ylabel(
            self.params.ylabel
        ) if self.params.ylabel is not None else ...

    def set_lims(self):
        self.ax.set_xlim(*self.params.xrange) if self.params.xrange is not None else ...
        self.ax.set_ylim(*self.params.yrange) if self.params.yrange is not None else ...

    def set_xticks(self):
        from matplotlib.ticker import AutoLocator, AutoMinorLocator

        self.ax.set_xticks(
            [float(i) for i in self.params.xticks.split()]
        ) if self.params.xticks is not None else self.ax.xaxis.set_major_locator(
            AutoLocator()
        )
        self.ax.set_xticklabels(
            self.params.xticklabels
        ) if self.params.xticklabels is not None else ...
        self.ax.xaxis.set_minor_locator(AutoMinorLocator())

    def set_yticks(self):
        from matplotlib.ticker import AutoLocator, AutoMinorLocator

        self.ax.set_yticks(
            [float(i) for i in self.params.yticks.split()]
        ) if self.params.yticks is not None else self.ax.yaxis.set_major_locator(
            AutoLocator()
        )
        self.ax.set_yticklabels(
            self.params.yticklabels
        ) if self.params.yticklabels is not None else ...
        self.ax.yaxis.set_minor_locator(AutoMinorLocator())

    def hide_ticks(self):
        if self.params.hide_xticks:
            self.ax.tick_params(which="both", bottom=False)
        if self.params.hide_yticks:
            self.ax.tick_params(which="both", left=False)


class HeatSet:
    def __init__(
        self,
        im,
        fig: "figure.Figure",
        ax: "plt.Axes",
        params: "HeatFigBase",
        vmin: float,
        vmax: float,
    ) -> None:
        import colorcet

        self.fig: figure.Figure = fig
        self.ax: plt.Axes = ax
        self.params: "HeatFigBase" = params
        self.vmin: float
        self.vmax: float
        if self.params.vrange is not None:
            self.vmin, self.vmax = self.params.vrange
        else:
            self.vmin, self.vmax = vmin, vmax
        self.vcenter: float = self.params.vcenter
        self.im = im

        ##
        self.set_norm()
        self.im.set_cmap(self.params.cmap)
        if self.params.colorbar:
            self.plot_colorbar()

    def set_norm(self):
        if self.params.norm == "Normal":
            norm = colors.Normalize(self.vmin, self.vmax)
        elif self.params.norm == "Logarithmic":
            if self.vmin < 0:
                raise ValueError(
                    "Logarithmic Normalization doesn't support negative values"
                )
            norm = colors.LogNorm(self.vmin, self.vmax)
        elif self.params.norm == "Centered":
            norm = colors.CenteredNorm(self.vcenter)
        elif self.params.norm == "SymmetricLogarithmic":
            norm = colors.SymLogNorm(
                linthresh=self.params.symlogparm[0],
                linscale=self.params.symlogparm[1],
                vmin=self.vmin,
                vmax=self.vmax,
            )
        elif self.params.norm == "TwoSlopeNorm":
            norm = colors.TwoSlopeNorm(self.vmin, self.vcenter, self.vmax)
        elif self.params.norm == "PowerLaw":
            norm = colors.PowerNorm(self.params.power, self.vmin, self.vmax)
        else:
            norm = MyCustomNormalize(self.vmin, self.vmax, self.vcenter)
        self.im.set_norm(norm)

    def plot_colorbar(self):
        cbar = self.fig.colorbar(self.im, ax=self.ax)
        if self.params.cticks is not None:
            cticks = [float(i) for i in self.params.cticks.split()]
        elif self.vmin < 0 and self.vmax > 0:
            cticks = [self.vmin, self.vmin / 2, 0, self.vmax / 2, self.vmax]
        else:
            cticks = [
                self.vmin,
                (self.vmin + self.vmax) * 1 / 4,
                (self.vmin + self.vmax) / 2,
                (self.vmin + self.vmax) * 3 / 4,
                self.vmax,
            ]

        if self.params.cticklabels is not None:
            cticklabels = self.params.cticklabels.split()

        else:
            cticklabels: list[str] = [f"{i:.2f}" for i in cticks]

        if len(cticks) != len(cticklabels):
            raise ValueError("cticks and cticklabels must have same length")

        cbar.set_ticks(cticks)
        cbar.set_ticklabels(cticklabels)

        cbar.ax.set_ylim(self.vmin, self.vmax)


class FigPlotBase:
    def __init__(self, params: "FigSetBase", fig: "figure.Figure", ax: "plt.Axes"): ...


def save_show(
    plot_cls: type[FigPlotBase],
    params: "FigSetBase",
):
    from importlib import resources

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if params.matplotlibrc.exists():
        mpl.rc_file(params.matplotlibrc)
    else:
        with resources.path("mxmftools", "hb.style") as rc_path:
            mpl.rc_file(rc_path)

    if params.from_cli is False:
        return (plot_cls, params)

    fig, ax = plt.subplots()

    plot_cls(params, fig, ax)
    for savefile in params.save.split():
        plt.savefig(savefile)
    if params.show:
        plt.show()


def plot_from_cli_str(str_params: str, fig, ax):
    import importlib
    import shlex

    import click
    from typer.main import get_command

    params_list = shlex.split(str_params)
    params_list.extend(["--from_cli", "--dontshow"])

    info_name = params_list[0]
    args = params_list[1:]
    app = importlib.import_module(f"mxmftools.{info_name}.cli").app

    print(params_list)

    cmd: click.Command = get_command(app)
    # print(cmd.make_context(info_name, args))
    # rv = get_command(app).invoke(cmd.make_context(info_name, args))
    rv = get_command(app).invoke(cmd.make_context(info_name, args))
    rv[0](rv[1], fig, ax)
