import numpy as np
from ..data_read import Readvaspout, ReadVasprun


def get_gap(ylist, kpoints, stdout=False):
    gap1, gap2 = 0, 0
    extremum = np.array(
        [[ylist[0, :, i].min(), ylist[0, :, i].max()] for i in range(ylist.shape[2])]
    )
    positive_extremum = (extremum + np.absolute(extremum)) / 2
    negative_extremum = (extremum - np.absolute(extremum)) / 2
    if ((extremum[:, 0] * extremum[:, 1]) > 0).all():
        vbm_band = negative_extremum[:, 1].nonzero()[0][-1]
        cbm_band = positive_extremum[:, 0].nonzero()[0][0]
    else:
        vbm_band = negative_extremum[:, 1].nonzero()[0][-1] + 1
        cbm_band = positive_extremum[:, 0].nonzero()[0][0] - 1
    vbm = ylist[0, :, vbm_band].max()
    cbm = ylist[0, :, cbm_band].min()
    vbm_index = np.array(np.where(ylist[0, :, vbm_band] == vbm))
    vbm_kpoints = list(set(tuple(tuple(kpoint) for kpoint in kpoints[vbm_index[0]])))
    cbm_index = np.array(np.where(ylist[0, :, cbm_band] == cbm))
    cbm_kpoints = list(set(tuple(tuple(kpoint) for kpoint in kpoints[cbm_index[0]])))

    if stdout:
        print("========== 1th spin========")
        print(f"vbm locates at {vbm_kpoints} of {vbm_band+1}th band")
        print(f"cbm locates at {cbm_kpoints} of {cbm_band+1}th band")
    if ((extremum[:, 0] * extremum[:, 1]) > 0).all():
        gap1 = cbm - vbm
        if stdout:
            print(f"gap is {gap1}")
    else:
        gap1 = 0
        if stdout:
            print("Metal")

    try:
        extremum = np.array(
            [
                [ylist[1:, :, i].min(), ylist[1:, :, i].max()]
                for i in range(ylist.shape[2])
            ]
        )
        positive_extremum = (extremum + np.absolute(extremum)) / 2
        negative_extremum = (extremum - np.absolute(extremum)) / 2
        if ((extremum[:, 0] * extremum[:, 1]) > 0).all():
            vbm_band = negative_extremum[:, 1].nonzero()[0][-1]
            cbm_band = positive_extremum[:, 0].nonzero()[0][0]
        else:
            vbm_band = negative_extremum[:, 1].nonzero()[0][-1] + 1
            cbm_band = positive_extremum[:, 0].nonzero()[0][0] - 1
        vbm = ylist[1, :, vbm_band].max()
        cbm = ylist[1, :, cbm_band].min()
        vbm_index = np.array(np.where(ylist[1, :, vbm_band] == vbm))
        vbm_kpoints = list(
            set(tuple(tuple(kpoint) for kpoint in kpoints[vbm_index[0]]))
        )
        cbm_index = np.array(np.where(ylist[1, :, cbm_band] == cbm))
        cbm_kpoints = list(
            set(tuple(tuple(kpoint) for kpoint in kpoints[cbm_index[0]]))
        )
        if stdout:
            print("========== 2th spin========")
            print(f"vbm locates at {vbm_kpoints} of {vbm_band+1}th band")
            print(f"cbm locates at {cbm_kpoints} of {cbm_band+1}th band")
        if ((extremum[:, 0] * extremum[:, 1]) > 0).all():
            gap2 = cbm - vbm
            if stdout:
                print(f"gap is {gap2}")
        else:
            gap2 = 0
            if stdout:
                print("Metal")
        return (gap1, gap2)
    except:
        return gap1


def gap_from_h5(file="vaspout.h5"):
    data = Readvaspout(file)
    ylist = data.eigenvalues - data.fermi
    gap = get_gap(ylist, data.kpoints)
    return gap


def gap_from_h5_opt(file="vaspout.h5"):
    data = Readvaspout(file)
    ylist = data.eigenvalues_opt - data.fermi
    gap = get_gap(ylist, data.kpoints_opt)
    return gap


def gap_from_xml(file="vasprun.xml"):
    data = ReadVasprun(file)
    ylist = data.eigenvalues - data.fermi
    gap = get_gap(ylist, data.kpoints)
    return gap


def gap_from_xml_opt(file="vasprun.xml"):
    data = ReadVasprun(file)
    ylist = data.eigenvalues_opt - data.fermi
    gap = get_gap(ylist, data.kpoints_opt)
    return gap
