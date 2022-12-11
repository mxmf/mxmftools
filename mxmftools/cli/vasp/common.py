import click
from ...utils import OptionEatAll

vasp_common_options = [
    click.option("-s", "--spin", type=int, help="specify spin chanel"),
    click.option(
        "-c",
        "--color",
        type=tuple,
        cls=OptionEatAll,
        mytype="strlist",
        help="specify color",
    ),
    click.option(
        "-yr",
        "--yrange",
        type=(float, float),
        default=[-4, 6],
        help="energy range to plot",
    ),
    click.option(
        "-a",
        "--atoms",
        cls=OptionEatAll,
        multiple=True,
        default=None,
        help="specify projected atoms",
    ),
    click.option(
        "-o",
        "--orbitals",
        cls=OptionEatAll,
        multiple=True,
        help="specify projected orbitals",
    ),
    click.option(
        "--save",
        type=tuple,
        cls=OptionEatAll,
        mytype="strlist",
        default=None,
        help="savefig as",
    ),
    click.option(
        "-vf",
        "--vaspfileformat",
        type=click.Choice(["hdf5", "h5", "xml"]),
        default="h5",
        required=False,
        envvar="MXMF_VASPFILE_FORMAT",
        help="read file format, you can specify it use envvar 'MXMF_VASPFILE_FORMAT'",
    ),
]


def common():
    print("common")
