# -*- coding: utf-8 -*-
import click
from .wannier90 import wannier90
from .vasp import vasp
from .vaspberry import vaspberry
from .phonopy import phonopy
from click_bash42_completion import patch

patch()


@click.group()
def main():
    pass


main.add_command(vasp)
main.add_command(wannier90)
main.add_command(vaspberry)
main.add_command(phonopy)
