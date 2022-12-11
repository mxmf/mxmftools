import click
from .curv import curv


@click.group()
def vaspberry():
    pass


vaspberry.add_command(curv)
