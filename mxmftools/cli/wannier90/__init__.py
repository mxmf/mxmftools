import click
from .band import band
from .compare import compare


@click.group()
def wannier90():
    pass


wannier90.add_command(band)
wannier90.add_command(compare)
