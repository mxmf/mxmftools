import click
from .band import band


@click.group()
def phonopy():
    pass


phonopy.add_command(band)
