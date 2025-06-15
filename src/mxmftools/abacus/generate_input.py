from ase import Atoms
from pathlib import Path


def write_stru(atoms: Atoms, stru_file: Path = Path("./STRU")):
    with open(stru_file, "w") as fstru:
        _ = fstru.write("ATOMIC_SPECIES\n")
        for symbol in set(atoms.symbols):
            _ = fstru.write(f"{symbol}\n")

    ...
