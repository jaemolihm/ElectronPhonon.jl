#!/bin/bash
set -e
export OMP_NUM_THREADS=1

# quantum espresso executable
QE=$HOME/program/qe-epw/bin

# SCF, phonon
mkdir -p dyn_dir
mpirun -np 16 $QE/pw.x -nk 4 -ndiag 1 -in scf.in > scf.out
mpirun -np 16 $QE/ph.x -nk 4 -ndiag 1 -in ph.in > ph.out
mpirun -np 1 $QE/q2r.x -in q2r.in > q2r.out
epw_pp.py bn
rm temp/_ph0/bn.q_*/bn.wfc*

# EPW setup
mpirun -np 16 $QE/pw.x -nk 4 -ndiag 1 -in scf.in > scf.out
mpirun -np 16 $QE/pw.x -nk 4 -ndiag 1 -in bands.in > bands.out
my_qe_bands.py bn temp
mpirun -np 16 $QE/pw.x -nk 4 -ndiag 1 -in nscf.in > nscf.out
mpirun -np 16 $QE/epw.x -nk 16 -in epw.setup.in > epw.setup.out

# EPW
mpirun -np 1 $QE/epw.x -nk 1 -in epw.selfen.in > epw.selfen.out
mpirun -np 1 $QE/epw.x -nk 1 -in epw.transport.hole.in > epw.transport.hole.out
rm inv_tau* restart.fmt IBTEvel_sup.fmt
mpirun -np 1 $QE/epw.x -nk 1 -in epw.transport.elec.in > epw.transport.elec.out
./parse_epwout.py
