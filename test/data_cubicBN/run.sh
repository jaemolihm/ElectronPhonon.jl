#!/bin/sh
#PBS -N bn.6
#PBS -q batch
#PBS -l nodes=2:G1:ppn=20
#PBS -l walltime=10:00:00
#PBS -e error
#PBS -o output
cd $PBS_O_WORKDIR
cat $PBS_NODEFILE > nodefile
NPROC=`wc -l < $PBS_NODEFILE`
# quantum espresso executable
QE=/home/jmlim/program/qe-dev/bin

# # SCF, phonon
# mkdir -p dyn_dir
# mpirun -np 40 -hostfile nodefile $QE/pw.x -nk 20 -ndiag 1 -in scf.in > scf.out
# mpirun -np 40 -hostfile nodefile $QE/ph.x -nk 8 -ndiag 1 -in ph.in > ph.out
# mpirun -np 1 -hostfile nodefile $QE/q2r.x -in q2r.in > q2r.out
# ~/bin/epw_pp.py bn
# rm temp/_ph0/bn.q_*/bn.wfc*

mpirun -np 40 -hostfile nodefile $QE/pw.x -nk 20 -ndiag 1 -in scf.in > scf.out
mpirun -np 40 -hostfile nodefile $QE/pw.x -nk 8 -ndiag 1 -in nscf.in > nscf.out

mpirun -np 8 -hostfile nodefile $QE/epw.x -nk 8 -in epw_setup.in > epw_setup.out
