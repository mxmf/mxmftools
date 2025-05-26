import typer
from .vasp_cli import app as vasp_app

app = typer.Typer(
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_enable=False,
)

app.add_typer(vasp_app, name="vasp")
