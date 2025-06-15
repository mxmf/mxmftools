import typer

from .abacus.cli import app as abacus_app
from .subfig import subfig
from .vasp.cli import app as vasp_app

app = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)

app.add_typer(vasp_app, name="vasp")
app.add_typer(abacus_app, name="abacus")
_ = app.command()(subfig)


# app.add_typer(subfig_app, name="subfig")
