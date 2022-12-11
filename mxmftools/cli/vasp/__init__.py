import click
from .band import band
from .dos import dos


@click.group()
def vasp():
    pass


vasp.add_command(band)
vasp.add_command(dos)
