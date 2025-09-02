from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
import typer
from mxmftools.vasp import BandParams


@dataclass
class CompareParams(BandParams):
    wannierfile: Annotated[Path, typer.Argument(exists=True)] = Path(
        "wannier90_band.dat"
    )
