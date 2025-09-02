from .lazy_imports import lazy_load_module_symbols
import typing as _t

if _t.TYPE_CHECKING:
    from .common_params import FigSetBase, HeatFigBase
    from .plot_utils import (
        AxesSet,
        FigPlotBase,
        plot_from_cli_str,
        set_style,
        HeatSet,
        plot_series,
    )

__all__ = [
    "FigSetBase",
    "HeatFigBase",
    "AxesSet",
    "FigPlotBase",
    "plot_from_cli_str",
    "set_style",
    "lazy_load_module_symbols",
    "HeatSet",
    "plot_series",
]

lazy_load_module_symbols(
    {
        "FigSetBase": ".common_params",
        "HeatFigBase": ".common_params",
        "AxesSet": ".plot_utils",
        "FigPlotBase": ".plot_utils",
        "plot_from_cli_str": ".plot_utils",
        "set_style": ".plot_utils",
        "lazy_load_module_symbols": ".plot_utils",
        "HeatSet": ".plot_utils",
        "plot_series": ".plot_utils",
    },
    globals(),
)
