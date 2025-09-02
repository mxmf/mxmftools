import typer

from .subfig import subfig
from .vasp.cli import app as vasp_app
from .wannier90.cli import app as wannier_app
from spinmc import app as spinmc_app  # type:ignore

app = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)

app.add_typer(vasp_app, name="vasp")
app.add_typer(wannier_app, name="wannier90")
app.add_typer(spinmc_app, name="spinmc")
_ = app.command()(subfig)
