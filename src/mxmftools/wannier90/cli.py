import typer
from . import CompareParams
from mxmftools.utils import dataclass_cli

app = typer.Typer(no_args_is_help=True)


@app.command("compare")
@dataclass_cli
def band(params: CompareParams):
    from . import ComparePlot
    from mxmftools.utils import plot_series

    plot_series(ComparePlot, params)
