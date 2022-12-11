# -*- coding: utf-8 -*-
import numpy as np
import h5py


class Readvaspout(object):
    def __init__(self, file="vaspout.h5"):
        self.file = h5py.File(file, "r")

    @property
    def labels_kpoints(self):
        try:
            labels = self.file["input/kpoints_opt/labels_kpoints"][:]
        except:
            labels = self.file["input/kpoints/labels_kpoints"][:]
        xticklabels = []
        for i, label in enumerate(labels):
            if i == 0 or label != labels[i - 1]:
                if label.decode("utf-8") not in [chr(i) for i in range(65, 91)] + [
                    chr(i) for i in range(97, 123)
                ]:
                    xticklabels.append(f"${label.decode('utf-8')}$")
                else:
                    xticklabels.append(label.decode("utf-8"))
        return xticklabels

    @property
    def number_kpoints(self):
        try:
            n = np.int32(self.file["input/kpoints_opt/number_kpoints"])
        except:
            n = np.int32(self.file["input/kpoints/number_kpoints"])
        return n

    @property
    def symbols(self):
        ion_types = self.file["results/positions/ion_types"][:]
        number_ion_types = self.file["results/positions/number_ion_types"][:]
        symbols = []
        for i, ion in enumerate(ion_types):
            symbols.extend([ion_types[i].decode("utf-8")] * number_ion_types[i])
        return symbols

    @property
    def nedos(self):
        nedos = np.int32(self.file["results/electron_dos/nedos"])
        return nedos

    @property
    def ionnum(self):
        ionnum = self.file["results/positions/number_ion_types"][:].sum()
        return ionnum

    @property
    def position(self):
        position = self.file["results/positions/position_ions"][:]
        return position

    @property
    def real_cell(self):
        real_cell = self.file["results/positions/lattice_vectors"][:]
        return real_cell

    @property
    def rec_cell(self):
        rec_cell = np.zeros((3, 3))
        rec_cell[:, 0] = np.cross(self.real_cell[:, 1], self.real_cell[:, 2]) / np.dot(
            np.cross(self.real_cell[:, 0], self.real_cell[:, 1]), self.real_cell[:, 2]
        )
        rec_cell[:, 1] = np.cross(self.real_cell[:, 2], self.real_cell[:, 0]) / np.dot(
            np.cross(self.real_cell[:, 0], self.real_cell[:, 1]), self.real_cell[:, 2]
        )
        rec_cell[:, 2] = np.cross(self.real_cell[:, 0], self.real_cell[:, 1]) / np.dot(
            np.cross(self.real_cell[:, 0], self.real_cell[:, 1]), self.real_cell[:, 2]
        )
        return rec_cell

    @property
    def fermi(self):
        fermi_energy = np.float64(self.file["results/electron_dos/efermi"])
        return fermi_energy

    @property
    def kpoints_opt(self):
        try:
            kpoints = self.file[
                "results/electron_eigenvalues_kpoints_opt/kpoint_coords"
            ][:]
            return kpoints
        except:
            return None

    @property
    def kpoints(self):
        kpoints = self.file["results/electron_eigenvalues/kpoint_coords"][:]
        return kpoints

    @property
    def weights_opt(self):
        try:
            weights = self.file[
                "results/electron_eigenvalues_kpoints_opt/kpoints_symmetry_weight"
            ][:]
            return weights
        except:
            return None

    @property
    def weights(self):
        weights = self.file["results/electron_eigenvalues/kpoints_symmetry_weight"][:]
        return weights

    @property
    def nbands(self):
        nbands = self.file["results/electron_eigenvalues/eigenvalues"].shape[-1]
        return nbands

    @property
    def eigenvalues_opt(self):
        try:
            eigenvalues = self.file[
                "results/electron_eigenvalues_kpoints_opt/eigenvalues"
            ][:]
            return eigenvalues
        except:
            return None

    @property
    def eigenvalues(self):
        eigenvalues = self.file["results/electron_eigenvalues/eigenvalues"][:]
        return eigenvalues

    @property
    def dos(self):
        dos = self.file["results/electron_dos/dos"][:]
        return dos

    @property
    def dosi(self):
        dosi = self.file["results/electron_dos/dosi"][:]
        return dosi

    @property
    def dospar(self):
        dospar = self.file["results/electron_dos/dospar"][:]
        return dospar

    @property
    def dose(self):
        dose = self.file["results/electron_dos/energies"][:]
        return dose

    @property
    def nedos(self):
        nedos = np.int32(self.file["results/electron_dos/nedos"][:])
        return nedos

    @property
    def projected_opt(self):
        try:
            projected = self.file["results/projectors_kpoints_opt/par"][:]
            return projected
        except:
            return None

    @property
    def projected(self):
        projected = self.file["results/projectors/par"][:]
        return projected

    @property
    def moments(self):
        moments = self.file["intermediate/ion_dynamics/magnetism/moments"][:]
        return moments


if __name__ == "__main__":
    data = Readvaspout("vaspout.h5")
