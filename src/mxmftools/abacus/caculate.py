import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict, final

from ase import Atoms
from ase.calculators.abacus import Abacus, AbacusProfile


class KptsFromAse(TypedDict):
    size: tuple[int, int, int] | None
    density: float | None
    gamma: None | bool
    even: None | bool


@dataclass
@final
class AbacusParams:
    def __init__(
        self,
        spintype: Literal["nospin", "spin", "spin+soc", "nc_spin", "nc_spin+soc"],
        gamma_only: Literal[0, 1, None] = None,
        pseudo_dir: str | None = os.getenv("ABACUS_PP_PATH"),
        orbital_dir: str | None = os.getenv("ABACUS_ORBITAL_PATH"),
        ecutwfc: float = 100,
        scf_nmax: int = 100,
        scf_thr: float = 1e-7,
        basis_type: Literal["pw", "lcao", "lcao_in_pw"] = "lcao",
        smearing_method: Literal["fixed", "gauss", "mp", "mp2", "mv", "fd"] = "gauss",
        smearing_sigma: float = 0.015,
        calculation: Literal[
            "relax",
            "scf",
            "nscf",
            "cell-relax",
            "md",
            "get_pchg",
            "get_wf",
            "get_S",
            "gen_bessel",
            "test_memory",
            "test_neighbour",
        ] = "scf",
        ks_solver: Literal[
            "cg",
            "bpcg",
            "dav",
            "dav_subspace",
            "lapack",
            "genelpa",
            "scalapack_gvx",
            "cusolver",
            "cusolvermp",
            "elpa",
            None,
        ] = None,
        symmetry: Literal[-1, 0, 1, None] = None,
        init_wfc: Literal["atomic", "file", "random", "nao+random"] = "atomic",
        init_chg: Literal["atomic", "file", "wfc", "auto"] = "atomic",
        # Parameter DFT+U
        dft_plus_u: Literal[0, 1, 2] = 0,
        orbital_corr: list[int] | None = None,
        hubbard_u: list[float] | None = None,
        out_chg: int = 1,
        # Parameter relax
        force_thr_ev: float = 0.0257112,  # unit eV/Angstrom ,recommended value for using atomic orbitals is 0.04 eV/Angstrom
        stress_thr: float = 0.5,  # unit kpar
        relax_method: Literal[
            "cg", "bfgs", "bfgs_trad", "cg_bfgs", "sd", "fire", ""
        ] = "cg",
        relax_nmax: int = 100,
        kpar: int = 1,
        # Parameter KPT file
        # kpts: tuple[int, int, int] | KptsFromAse = (9, 9, 1),
        kpts: tuple[int, int, int] | list[tuple[float, float, float]] = (9, 9, 1),
        koffset: tuple[int, int, int] | int = (0, 0, 0),
        # "Gamma, MP, Direct, Cartesian, or Line."
        kmode: Literal["Gamma", "MP", "Cartesian", "Direct", "Line"] = "Gamma",
        knumbers: list[int] | None = None,
        vdw_method: Literal["d2", "d3_0", "d3_bj", "none"] = "none",
    ) -> None:
        local_vars = locals()
        exclude_keys = ["orbital_corr", "hubbard_u", "spintype", "self"]

        self.result = {k: v for k, v in local_vars.items() if k not in exclude_keys}

        self.result |= {
            "orbital_corr": " ".join(str(i) for i in orbital_corr)
            if orbital_corr is not None
            else None
        }

        self.result |= {
            "hubbard_u": " ".join(str(i) for i in hubbard_u)
            if hubbard_u is not None
            else None
        }

        # self.result |= dict(zip(_spin_keys, self.set_spin(spintype)))
        self.result |= dict(
            zip(("nspin", "lspinorb", "noncolin"), self.set_spin(spintype))
        )

        self.pre_check()

    def set_spin(
        self, spintype: Literal["nospin", "spin", "spin+soc", "nc_spin", "nc_spin+soc"]
    ) -> tuple[int, int, int]:
        match spintype:
            case "nospin":
                return (1, 0, 0)
            case "spin":
                return (2, 0, 0)
            case "spin+soc":
                return (4, 1, 0)
            case "nc_spin":
                return (4, 0, 1)
            case "nc_spin+soc":
                return (4, 1, 1)

    def pre_check(self):
        basis_type = self.result.get("basis_type")
        ks_solver = self.result.get("ks_solver")
        if (basis_type == "lcao") and (
            ks_solver in ["cg", "bpcg", "dav", "dav_subspace"]
        ):
            raise Exception(f"{basis_type}不能使用{ks_solver}求解器")

        elif (basis_type == "pw") and (
            ks_solver
            in [
                "lapack",
                "genelpa",
                "scalapack_gvx",
                "cusolver",
                "cusolvermp",
                "elpa",
            ]
        ):
            raise Exception(f"{basis_type}不能使用{ks_solver}求解器")

        calculation = self.result.get("calculation")
        kmode = self.result.get("kmode")

        if calculation == "scf" and kmode == "Line":
            raise Exception(f"{calculation}计算不能使用{kmode}模式生成KPT文件")


def abacusrun(PREFIX: str, directory: Path):
    command = os.getenv("ASE_ABACUS_COMMAND")
    if command is None:
        raise ValueError
    else:
        command = command.replace("PREFIX", PREFIX)
    with open(f"{directory}/abacus.out", "w") as outfile:
        errorcode = subprocess.run(
            command, shell=True, check=True, cwd=directory, stdout=outfile
        )
    print(errorcode)


def generate_input_files(atoms: Atoms, params: AbacusParams, directory: Path):
    calc = Abacus(
        profile=AbacusProfile(command="mpirun -n 4 abacus"),
        directory=str(directory),
        **params.result,
    )
    calc.write_inputfiles(atoms, "energy")


def do_abacus_calc(
    atoms: Atoms, params: AbacusParams, directory: Path, PREFIX: str = ""
):
    generate_input_files(atoms, params, directory)
    abacusrun(PREFIX, directory)
