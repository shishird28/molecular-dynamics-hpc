# molecular-dynamics-hpc
Ewald-based methods for modelling long-range electrostatics. Includes serial direct Ewald and PME with HPC methods using MPI to simulate the dynamics of a NaCl crystal.

## Features

- Molecular dynamics simulation engine written in C++
- MPI parallelisation for distributed-memory HPC systems
- Velocity Verlet integration
- Periodic boundary conditions
- Ewald / PME electrostatics
- Energy conservation diagnostics

---

## Compilation

Add to ~/.bashrc:
```bash
module load fftw/3.3.10-gcc-8.5.0-ztz2p3t
export FFTW_PREFIX="/home/support/rl8/spack/0.21.2/spack/opt/spack/linux-rocky8-ivybridge/gcc-8.5.0/fftw-3.3.10-ztz2p3tws3rqxpithqvazalyg2qvjudw"
```
Compile using MPI:
```bash
mpicxx filename.cpp -o filename -I${FFTW_PREFIX}/include -L${FFTW_PREFIX}//lib -lfftw3_mpi -lfftw3 -lm
```
---

### Local Run

```bash
mpirun -np 32 ./filename
```

## Outputs

- PME_traj_v1.cpp & direct_ewald_vfinal.cpp: .xyz file of ion positions at each time step allowing visualisation of trajectories in MD visualisers such as VMD.
- PME_traj_energydrift.cpp: .dat file plotting fractional energy drift from thermalisation point against time.
- PME_traj_VCF.cpp: .dat file plotting Velocity Autocorrelation functions of Na and Cl seperately.
