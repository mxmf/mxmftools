from mxmftools.utils import lazy_load_module_symbols

import typing as _t

if _t.TYPE_CHECKING:
    from .band.params import BandParams
    from .dos.params import DosParams
    from .band.bandplot import BandPlot, BandData
    from .dos.dosplot import DosPlot
    from .dataread import ReadVaspout, ReadVasprun
    from .vasp_utils import get_gap

__all__ = [
    "BandParams",
    "DosParams",
    "BandPlot",
    "BandData",
    "DosPlot",
    "ReadVaspout",
    "ReadVasprun",
    "get_gap",
]

lazy_load_module_symbols(
    {
        "BandParams": ".band.params",
        "DosParams": ".dos.params",
        "BandPlot": ".band.bandplot",
        "BandData": ".band.bandplot",
        "DosPlot": ".dos.dosplot",
        "ReadVaspout": ".dataread",
        "ReadVasprun": ".dataread",
        "get_gap": ".vasp_utils",
    },
    globals(),
)
