#!/usr/bin/env python3
import numpy as np

ry2ev = 13.60569253

def read_epw_out_ph(filename, nq, nmodes, ntemperatures):
    # NOTE: Multiple smearing is not implemented.
    energy_phonon = np.zeros((nq, nmodes))
    gamma = np.zeros((ntemperatures, nq, nmodes))

    iq = 0
    itemp = 0
    f = open(filename, 'r')
    while True:
        line = f.readline()
        if 'ismear' in line:
            line = f.readline()
            for imode in range(nmodes):
                data = f.readline().split()
                f.readline()
                gamma[itemp, iq, imode] = float(data[-5])
                energy_phonon[iq, imode] = float(data[-2])

            if itemp < ntemperatures - 1:
                itemp += 1
            else:
                itemp = 0
                iq += 1

        if not line:
            break

    f.close()

    energy_phonon /= 1000.0 # meV to eV
    gamma /= 1000.0 # meV to eV

    return energy_phonon, gamma

def read_epw_out_el(filename, nkftot, nband, ntemperatures):
    energy = np.zeros((nkftot, nband))
    selfen = np.zeros((ntemperatures, nkftot, nband), dtype=complex)

    # Parse Fermi energy
    efermi = None
    with open(filename, 'r') as f:
        for line in f:
            if "Fermi energy is calculated from the fine k-mesh" in line:
                efermi = float(line.split()[-2])
                break
            if "Fermi energy is read from the input file" in line:
                efermi = float(line.split()[-2])
                break
    if efermi is None:
        raise ValueError("Fermi energy not found")

    with open(filename, 'r') as f:
        for itemp in range(ntemperatures):
            while True:
                line = f.readline()
                if line.strip() == "WARNING: only the eigenstates within the Fermi window are meaningful":
                    break
            f.readline()
            for ik in range(nkftot):
                f.readline()
                f.readline()
                for ib in range(nband):
                    data = f.readline().split()
                    data_slim = [data[3], data[6], data[9], data[12], data[14]]
                    energy[ik, ib] = float(data_slim[0])
                    selfen[itemp, ik, ib] = float(data_slim[1]) + 1j * float(data_slim[2])
                f.readline()
                f.readline()
                f.readline()

    energy += efermi
    selfen /= 1000.0 # meV to eV

    return energy, selfen



def read_specfun_ph(nq, nmodes, ntemperatures):
    data = np.loadtxt("specfun_sup.phon")
    nomega = data.shape[0] // (nq * nmodes * ntemperatures)
    Tlist = data[::nmodes*nomega, 2][:ntemperatures]
    omegalist = data[::nmodes, 4][:nomega]

    # index: (iq, itemperature, iomega, imode)
    selfen = (data[:, 5] + 1j * data[:, 7]).reshape((nq, ntemperatures, nomega, nmodes))
    # change index to (iomega, imode, itemperature, iq)
    selfen = np.transpose(selfen, (2, 3, 1, 0))

    specfun = np.zeros((nomega, ntemperatures, nq))
    for iT in range(ntemperatures):
        filename = f"specfun.phon.{Tlist[iT]:.3f}K"
        # index: (iq, iomega)
        data = np.loadtxt(filename)[:,2].reshape((nq, nomega))
        # change index to (iomega, iq)
        data = data.transpose()
        specfun[:, iT, :] = data

    selfen /= 1000.0 # meV to eV
    specfun *= 1000.0 # meV to eV. Unit of specfun is [1/energy].

    return selfen, specfun

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Parse self-energy calculation
    filename = "epw.selfen.out"
    nk = 125
    nq = 125
    nband = 7
    nmodes = 6
    ntemperatures = 2

    el_energy, el_imsigma = read_epw_out_el(filename, nk, nband, ntemperatures)
    ph_energy, ph_imsigma = read_epw_out_ph(filename, nq, nmodes, ntemperatures)
    ph_selfen_omega, ph_specfun = read_specfun_ph(nq, nmodes, ntemperatures)

    # Save transpose array because Julia uses column-major order while
    # python uses row-major order.
    np.save("el_energy.npy", np.transpose(el_energy))
    np.save("el_imsigma.npy", np.transpose(el_imsigma))
    np.save("ph_energy.npy", np.transpose(ph_energy))
    np.save("ph_imsigma.npy", np.transpose(ph_imsigma))

    # ph_selfen_omega, ph_specfun are already in column-major order. No need to transpose.
    # np.save("ph_selfen_omega.npy", ph_selfen_omega)
    # np.save("ph_specfun.npy", ph_specfun)
