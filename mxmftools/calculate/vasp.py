from ase.calculators.vasp import Vasp
from ase.calculators.vasp.create_input import GenerateVaspInput
from ase.calculators.vasp.create_input import list_int_keys, list_float_keys
from ase.io import read
import pathlib
import shutil
import h5py
import os
import subprocess
import numpy as np


def vasprun(PREFIX, directory):
    command = os.getenv("ASE_VASP_COMMAND").replace("PREFIX", PREFIX)
    # test = subprocess.run(command,shell=True, check=True)
    with open(f"{directory}vasp.out", "w") as outfile:
        errorcode = subprocess.run(
            command, shell=True, check=True, cwd=directory, stdout=outfile
        )
    print(errorcode)


list_int_keys.append("ioptcell")
list_float_keys.append("magmom")


default_calcdic = {
    "encut": 500,
    "setups": "recommended",
    "kpts": (12, 12, 1),
    "ncore": 4,
    "xc": "PBE",
    "prec": "Accurate",
    "algo": "normal",
    "gamma": True,
    "sigma": 0.05,
    "ismear": -5,
    "ispin": 2,
    "nelm": 200,
    "ediff": 1e-6,
    "ediffg": -0.01,
    "isym": 2,
    "lorbit": 11,
    "lwave": False,
    "lreal": "Auto",
}


def write_kpoints_opt(path, dir=".", num=40):
    if dir[-1] == "/":
        file = "{}KPOINTS_OPT".format(dir)
    else:
        file = "{}/KPOINTS_OPT".format(dir)
    with open(file, "w") as fk:
        fk.write("KPOINT created by mxmf\n{}\nLine-mode\nreciprocal\n".format(num))
        for index, kpoint in enumerate(path[:-2]):
            fk.write(
                "    {:.6f}    {:.6f}    {:.6f}     {}\n".format(
                    kpoint[0], kpoint[1], kpoint[2], path[-1][index]
                )
            )
            fk.write(
                "    {:.6f}    {:.6f}    {:.6f}     {}\n".format(
                    path[index + 1][0],
                    path[index + 1][1],
                    path[index + 1][2],
                    path[-1][index + 1],
                )
            )
            fk.write("\n")
    return None


def write_kpoints(path, dir=".", num=40):
    if dir[-1] == "/":
        file = "{}KPOINTS".format(dir)
    else:
        file = "{}/KPOINTS".format(dir)
    with open(file, "w") as fk:
        fk.write("KPOINT created by mxmf\n{}\nLine-mode\nreciprocal\n".format(num))
        for index, kpoint in enumerate(path[:-2]):
            fk.write(
                "    {:.6f}    {:.6f}    {:.6f}     {}\n".format(
                    kpoint[0], kpoint[1], kpoint[2], path[-1][index]
                )
            )
            fk.write(
                "    {:.6f}    {:.6f}    {:.6f}     {}\n".format(
                    path[index + 1][0],
                    path[index + 1][1],
                    path[index + 1][2],
                    path[-1][index + 1],
                )
            )
            fk.write("\n")
    return None


def isp_relax(atoms, calcdic, calcdir="isp-relax"):
    """isp_relax

    Parameters
    ----------
    atoms : ase atoms

    calcdic : dictionary
        dictionary of calculate parameters
    calcdir : str, optional
        directary of calculate, by default "isp-relax"

    Returns
    -------
    _type_
        ase atoms
    """
    relax_dict = {
        "nsw": 100,
        "ibrion": 2,
        "isif": 3,
    }
    if calcdir[-1] == "/":
        pass
    else:
        calcdir += "/"
    dic = dict(dict(default_calcdic, **relax_dict), **calcdic)
    dic["ismear"] = 0
    ase_input_gener = GenerateVaspInput()
    ase_input_gener.set(**dic)
    ase_input_gener.initialize(atoms)
    try:
        pathlib.Path(calcdir).mkdir(
            parents=True,
        )
    except FileExistsError:
        pass
    ase_input_gener.write_incar(atoms, calcdir)
    ase_input_gener.write_potcar(directory=calcdir)
    ase_input_gener.write_kpoints(atoms, calcdir)
    atoms.write(f"{calcdir}/POSCAR")
    vasprun("vasp_std", calcdir)
    new_atoms = read(f"{calcdir}CONTCAR")
    return new_atoms


def isp_scf(atoms, calcdic, calcdir="isp-scf", path=None):
    """isp_scf

    Parameters
    ----------
    atoms : ase class atoms

    calcdic : dictionary
        dictionary of calculate parameters
    calcdir : str, optional
        directary of calculate, by default "isp-scf"
    path:
        use KPOINTS_OPT  to caculate band

    Returns
    -------
    _type_
        array of [free energy, energy without entropy, energy(sigma -> 0)] for every step
    """
    if calcdir[-1] == "/":
        pass
    else:
        calcdir += "/"
    dic = dict(default_calcdic, **calcdic)
    ase_input_gener = GenerateVaspInput()
    ase_input_gener.set(**dic)
    ase_input_gener.initialize(atoms)
    try:
        pathlib.Path(calcdir).mkdir(
            parents=True,
        )
    except FileExistsError:
        pass
    ase_input_gener.write_incar(atoms, calcdir)
    ase_input_gener.write_potcar(directory=calcdir)
    ase_input_gener.write_kpoints(atoms, calcdir)
    atoms.write(f"{calcdir}POSCAR")
    if path:
        write_kpoints_opt(path, calcdir)
    vasprun("vasp_std", calcdir)
    data = h5py.File(f"{calcdir}vaspout.h5")
    e = data["intermediate/ion_dynamics/energies"][:]
    return e


def isp_band(
    atoms, calcdic, prev_dir="isp-scf", calcdir="isp-band", path=None, line=True, num=50
):
    """isp-band

    Parameters
    ----------
    atoms : ase class atoms

    calcdic : dictionary
        dictionary of calculate parameters
    calcdir : str, optional
        directary of calculate, by default "isp-band"


    prev_dir :  str, optional
        isp_scf directory

    path:
        use KPOINTS_OPT  to caculate band
    line:
        whether use line mode, by default True
    num:
        num of kpoints in line mode, by default 50
    Returns
    -------
    _type_
        array of [free energy, energy without entropy, energy(sigma -> 0)] for every step
    """
    if prev_dir:
        shutil.copytree(pathlib.Path(prev_dir), pathlib.Path(calcdir))
    else:
        pass

    isp_band_dict = {
        "istart": 0,
        "icharg": 11,
        "ismear": 0,
    }
    if calcdir[-1] == "/":
        pass
    else:
        calcdir += "/"
    dic = dict(default_calcdic, **calcdic)
    dic = dict(dic, **isp_band_dict)
    ase_input_gener = GenerateVaspInput()
    ase_input_gener.set(**dic)
    ase_input_gener.initialize(atoms)
    try:
        pathlib.Path(calcdir).mkdir(
            parents=True,
        )
    except FileExistsError:
        pass
    ase_input_gener.write_incar(atoms, calcdir)
    ase_input_gener.write_potcar(directory=calcdir)
    if line:
        write_kpoints(path, calcdir, num)
    else:
        np.savetxt(
            f"{calcdir}KPOINTS",
            path,
            fmt="%2.6f",
            delimiter="\t",
            header=f"KPOINTS created by mxmf\n{len(path)}\nReciprocal lattice",
            comments="",
        )
    atoms.write(f"{calcdir}/POSCAR")
    vasprun("vasp_std", calcdir)
    data = h5py.File(f"{calcdir}vaspout.h5")
    e = data["intermediate/ion_dynamics/energies"][:]
    return e


def soc_scf(
    atoms, calcdic, prev_dir="isp-scf", calcdir="soc-scf", saxis=[0, 0, 1], path=None
):
    """soc_scf

    Parameters
    ----------
    atoms : ase class atoms

    calcdic : dictionary
        dictionary of calculate parameters
    calcdir : str, optional
        directary of calculate, by default "soc-scf"

    saxis : list, optional
        axis of magmom

    prev_dir :  str, optional
        isp_scf directory

    path:
        use KPOINTS_OPT  to caculate band

    Returns
    -------
    _type_
        array of [free energy, energy without entropy, energy(sigma -> 0)] for every step
    """
    if prev_dir:
        shutil.copytree(pathlib.Path(prev_dir), pathlib.Path(calcdir))
    else:
        pass

    soc_dict = {
        "istart": 0,
        "icharg": 1,
        # ismear:0,
        "lorbmom": True,
        "gga_compat": False,
        "lnoncollinear": True,
        "lsorbit": True,
        "saxis": saxis,
    }
    try:
        calcdic.pop("magmom")
    except KeyError:
        pass
    if calcdir[-1] == "/":
        pass
    else:
        calcdir += "/"
    calcdic["ismear"] = 0
    dic = dict(default_calcdic, **calcdic)
    dic = dict(dic, **soc_dict)
    ase_input_gener = GenerateVaspInput()
    ase_input_gener.set(**dic)
    ase_input_gener.initialize(atoms)
    try:
        pathlib.Path(calcdir).mkdir(
            parents=True,
        )
    except FileExistsError:
        pass
    ase_input_gener.write_incar(atoms, calcdir)
    ase_input_gener.write_potcar(directory=calcdir)
    ase_input_gener.write_kpoints(atoms, calcdir)
    atoms.write(f"{calcdir}/POSCAR")
    if path:
        write_kpoints_opt(path, calcdir)
    vasprun("vasp_ncl", calcdir)
    data = h5py.File(f"{calcdir}vaspout.h5")
    e = data["intermediate/ion_dynamics/energies"][:]
    return e


def soc_e(
    atoms, calcdic, prev_dir="isp-scf", calcdir="soc-scf", saxis=[0, 0, 1], path=None
):
    """soc_scf

    Parameters
    ----------
    atoms : ase class atoms

    calcdic : dictionary
        dictionary of calculate parameters
    calcdir : str, optional
        directary of calculate, by default "soc-scf"

    saxis : list, optional
        axis of magmom

    prev_dir :  str, optional
        isp_scf directory

    path:
        use KPOINTS_OPT  to caculate band

    Returns
    -------
    _type_
        array of [free energy, energy without entropy, energy(sigma -> 0)] for every step
    """
    if prev_dir:
        shutil.copytree(pathlib.Path(prev_dir), pathlib.Path(calcdir))
    else:
        pass

    soc_dict = {
        "istart": 0,
        "icharg": 11,
        # ismear:0,
        "lorbmom": True,
        "gga_compat": False,
        "lnoncollinear": True,
        "lsorbit": True,
        "saxis": saxis,
    }
    try:
        magmom = calcdic.pop("magmom")
        atoms.set_initial_magnetic_moments = magmom
    except KeyError:
        magmom = None
    # atoms.set_initial_magnetic_moments = calcdic["magmom"]
    if calcdir[-1] == "/":
        pass
    else:
        calcdir += "/"
    dic = dict(default_calcdic, **calcdic)
    dic = dict(dic, **soc_dict)
    ase_input_gener = GenerateVaspInput()
    ase_input_gener.set(**dic)
    ase_input_gener.initialize(atoms)
    try:
        pathlib.Path(calcdir).mkdir(
            parents=True,
        )
    except FileExistsError:
        pass
    ase_input_gener.write_incar(atoms, calcdir)
    if magmom:
        with open(f"{calcdir}INCAR", "a") as fi:
            fi.write("MAGMOM = ")
            [fi.write(f"{i} ") for i in magmom]
            fi.write("\n")
    ase_input_gener.write_potcar(directory=calcdir)
    ase_input_gener.write_kpoints(atoms, calcdir)
    atoms.write(f"{calcdir}/POSCAR")
    if path:
        write_kpoints_opt(path, calcdir)
    vasprun("vasp_ncl", calcdir)
    data = h5py.File(f"{calcdir}vaspout.h5")
    e = data["intermediate/ion_dynamics/energies"][:]
    return e


def soc_band(
    atoms,
    calcdic,
    prev_dir="soc-scf",
    calcdir="soc-band",
    saxis=[0, 0, 1],
    path=None,
    line=True,
    num=50,
):
    """soc_scf

    Parameters
    ----------
    atoms : ase class atoms

    calcdic : dictionary
        dictionary of calculate parameters
    calcdir : str, optional
        directary of calculate, by default "soc-scf"

    saxis : list, optional
        axis of magmom

    prev_dir :  str, optional
        isp_scf directory

    path:
        use KPOINTS_OPT  to caculate band

    num:
        band num, by default is 50
    Returns
    -------
    _type_
        array of [free energy, energy without entropy, energy(sigma -> 0)] for every step
    """
    if prev_dir:
        shutil.copytree(pathlib.Path(prev_dir), pathlib.Path(calcdir))
    else:
        pass

    soc_dict = {
        "istart": 0,
        "icharg": 11,
        "ismear": 0,
        "lorbmom": True,
        "gga_compat": False,
        "lnoncollinear": True,
        "lsorbit": True,
        "saxis": saxis,
    }
    try:
        magmom = calcdic.pop("magmom")
        atoms.set_initial_magnetic_moments = magmom
    except KeyError:
        magmom = None
    if calcdir[-1] == "/":
        pass
    else:
        calcdir += "/"
    dic = dict(default_calcdic, **calcdic)
    dic = dict(dic, **soc_dict)
    dic["ismear"] = 0
    ase_input_gener = GenerateVaspInput()
    ase_input_gener.set(**dic)
    ase_input_gener.initialize(atoms)
    try:
        pathlib.Path(calcdir).mkdir(
            parents=True,
        )
    except FileExistsError:
        pass
    ase_input_gener.write_incar(atoms, calcdir)
    if magmom:
        with open(f"{calcdir}INCAR", "a") as fi:
            fi.write("MAGMOM = ")
            [fi.write(f"{i} ") for i in magmom]
            fi.write("\n")
    ase_input_gener.write_potcar(directory=calcdir)
    if line:
        write_kpoints(path, calcdir, num)
    else:
        np.savetxt(
            f"{calcdir}KPOINTS",
            path,
            fmt="%2.6f",
            delimiter="\t",
            header=f"KPOINTS created by mxmf\n{len(path)}\nReciprocal lattice",
            comments="",
        )
    atoms.write(f"{calcdir}/POSCAR")
    vasprun("vasp_ncl", calcdir)
    data = h5py.File(f"{calcdir}vaspout.h5")
    e = data["intermediate/ion_dynamics/energies"][:]
    return e
