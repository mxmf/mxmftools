import typing as _t
from mxmftools.utils import lazy_load_module_symbols

if _t.TYPE_CHECKING:
    from .compare.params import CompareParams
    from .compare.plot import ComparePlot
__all__ = ["CompareParams", "ComparePlot"]

lazy_load_module_symbols(
    {
        "CompareParams": ".compare.params",
        "ComparePlot": ".compare.plot",
    },
    globals(),
)
