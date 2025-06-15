from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True)


@app.command("stru_from")
def get_stru_from_file(
    input_file: Annotated[Path, typer.Argument(exists=True)] = Path("POSCAR"),
    out_file: Path = Path("STRU"),
):
    pass

    from ase.io import read
    from ase.io.abacus import write_abacus
    from ase import Atoms

    atoms = read(input_file)
    assert isinstance(atoms, Atoms)  # ✅

    _ = write_abacus(out_file, atoms)


@app.command(name="down_pps")
def down_pps():
    import tempfile

    import pooch

    config_dir = Path.home() / ".config/mxmftools"

    _ = pooch.retrieve(
        url="http://www.quantum-simulation.org/potentials/sg15_oncv/sg15_oncv_upf_2020-02-06.tar.gz",
        known_hash="3f3bd74aa5d6e0b038218a6051bb99ed9469dc03d0f05b3ec8a523f0f7a7dff0",
        path=Path(tempfile.gettempdir()) / "mxmftools" / "pp",
        processor=pooch.Untar(
            extract_dir=config_dir / "pp" / "sg15_oncv_upf_2020-02-06"
        ),
    )


@app.command(name="down_orbitals")
def down_orbitals():
    import tempfile

    import pooch

    config_dir = Path.home() / ".config/mxmftools"
    _ = pooch.retrieve(
        url="https://abacus.ustc.edu.cn/_upload/tpl/0c/d8/3288/template3288/download/Libs/SG15-Version1p0__StandardOrbitals-Version2p0.zip",
        known_hash="141338cf7e9d58a5e8735df9124533c79b2a7945c08983e9f4d4cd6effa67edd",
        path=Path(tempfile.gettempdir()) / "mxmftools" / "orbitals",
        processor=pooch.Unzip(extract_dir=config_dir / "orbitals"),
    )
    _ = pooch.retrieve(
        url="https://abacus.ustc.edu.cn/_upload/tpl/0c/d8/3288/template3288/download/Libs/SG15-Version1p0__AllOrbitals-Version2p0.zip",
        known_hash="ce88a2de430125b6feeaa84f8f713a5cdc7064142367efc5eccb5ac840ce8432",
        path=Path(tempfile.gettempdir()) / "mxmftools" / "orbitals",
        processor=pooch.Unzip(extract_dir=config_dir / "orbitals"),
    )


@app.command(name="generate_calc_script")
def generate_calc_script():
    from importlib.resources import files, as_file
    import shutil

    template_file = files("mxmftools.abacus") / "caculate_template.py"
    with as_file(template_file) as src_path:
        _ = shutil.copy(src_path, Path("./cal.py"))
