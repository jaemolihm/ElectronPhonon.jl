#!/bin/bash
set -e
export OMP_NUM_THREADS=1

# quantum espresso executable
QE=/home/jmlim/julia_epw/qe-dev/bin
QE_ORIGINAL=/home/jmlim/julia_epw/qe-original/bin

# SCF, phonon
mkdir -p dyn_dir
mpirun -np 2 $QE/pw.x -nk 2 -ndiag 1 -in scf.in > scf.out
mpirun -np 2 $QE/ph.x -nk 2 -ndiag 1 -in ph.in > ph.out
mpirun -np 1 $QE/q2r.x -in q2r.in > q2r.out
~/bin/epw_pp.py bn
rm temp/_ph0/bn.q_*/bn.wfc*

# EPW setup
mpirun -np 2 $QE/pw.x -nk 2 -ndiag 1 -in scf.in > scf.out
mpirun -np 2 $QE/pw.x -nk 2 -ndiag 1 -in bands.in > bands.out
~/bin/my_qe_bands.py bn temp
mpirun -np 2 $QE/pw.x -nk 2 -ndiag 1 -in nscf.in > nscf.out
mpirun -np 2 $QE/epw.x -nk 2 -in epw_setup.in > epw_setup.out

# EPW
mpirun -np 1 $QE/epw.x -nk 1 -in epw.selfen.in > epw.selfen.out
mpirun -np 1 $QE/epw.x -nk 1 -in epw.transport.hole.in > epw.transport.hole.out
rm inv_tau* restart.fmt IBTEvel_sup.fmt
# NOTE: This must be run using QE_ORIGINAL because mp_mesh_k is not implemented in QE.
mpirun -np 1 $QE_ORIGINAL/epw.x -nk 1 -in epw.transport.elec.in > epw.transport.elec.out
./parse_epwout.py
