#!/bin/bash
set -e
export OMP_NUM_THREADS=1

# quantum espresso executable
QE=/home/jmlim/program/qe-dev/bin
QEJULIA=/home/jmlim/julia_epw/qe-dev/bin

## SCF, phonon
#mkdir -p dyn_dir
#mpirun -np 2 $QE/pw.x -nk 2 -ndiag 1 -in scf.in > scf.out
#mpirun -np 2 $QE/ph.x -nk 2 -ndiag 1 -in ph.in > ph.out
#mpirun -np 1 $QE/q2r.x -in q2r.in > q2r.out
#~/bin/epw_pp.py pb
#rm temp/_ph0/pb.q_*/pb.wfc*

## EPW setup
#mpirun -np 2 $QE/pw.x -nk 2 -ndiag 1 -in scf.in > scf.out
#mpirun -np 2 $QE/pw.x -nk 2 -ndiag 1 -in nscf.in > nscf.out
#mpirun -np 2 $QEJULIA/epw.x -nk 2 -in epw_setup.in > epw_setup.out

# EPW
mpirun -np 1 $QE/epw.x -nk 1 -in epw.transport.in > epw.transport.out
rm inv_tau* restart.fmt IBTEvel_sup.fmt sparse*
