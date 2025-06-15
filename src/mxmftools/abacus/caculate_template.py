from pathlib import Path

from ase import Atoms
from ase.io.abacus import read_abacus
from ase.io.vasp import read_vasp

from mxmftools.abacus.caculate import AbacusParams, do_abacus_calc

# atoms = read_vasp("./POSCAR")

# atoms.set_initial_magnetic_moments(None)
# atoms.set_initial_magnetic_moments([[1.0], [1.0], [1.0], [1.0]])
# atoms.set_initial_magnetic_moments(None)
# atoms.set_initial_magnetic_moments([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])

atoms: Atoms = read_abacus("./STRU")
atoms.set_initial_magnetic_moments(None)
atoms.set_initial_magnetic_moments([[1.0], [1.0], [1.0], [1.0]])
# atoms.set_initial_magnetic_moments(None)
# atoms.set_initial_magnetic_moments([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])


calc_params = AbacusParams(
    calculation="scf",
    spintype="spin",
    # dft+u
    dft_plus_u=1,
    orbital_corr=[2, 2, 0, 0],
    hubbard_u=[2.0, 2.0, 0, 0],
    # ---------------------------------
    # symmetry
    symmetry=None,
    # ---------------------------------
    # KPT SCF
    kmode="Gamma",
    kpts=(9, 9, 1),
    koffset=(0, 0, 0),
    # ---------------------------------
    # KPT Line mode
    # kmode="Line",
    # kpts=[(0.5, 0, 0), (1 / 3, 1 / 3, 0), (0, 0, 0), (0.5, 0, 0)],
    # knumbers=[30, 30, 30, 1],
)

do_abacus_calc(atoms, calc_params, Path("./test"))
