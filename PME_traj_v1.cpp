#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cassert>
#include <array>
#include <mpi.h>
#include <fftw3-mpi.h>
using namespace std;

// trajectory file
string filename = "PME_traj_scale.xyz";

/*
 * Class: ewald
 * ----------------------
 * Stores all parameters, cell data and functions for PME
 *
 */
class ewald {
    public:
    // physical parameters
    double alpha = 0.5; // convergence parameter
    int N_cell = 6; // number of unit cells per dimension
	int gspace = 4; // no. of FFT gridpoints between each ion
	ptrdiff_t N = 2*N_cell*gspace; 
	ptrdiff_t N0 = N, N1 = N, N2 = N;
	double a0 = 0.529177;      // Bohr radius in angstrom
	double a = 5.64 / a0;      // spacing constant of NaCl
	double L = a*N_cell;       // Length of cell
    double ds = L / N;         // FFT spacing
    double V_cell = pow(L,3);  // Volume of cell
	double rc = 1.5*a;         // real space cutoff
    double k = 1;              // Coulomb constant
	double charge = .885;      // charge magnitude
	double Cl_mass = 65092;    // Cl mass in au
	double Na_mass = 42213;    // Na mass in au
	
	// precomputed factors
	double pref = 1.0 / (V_cell*M_PI);
    double pre_alpha = (M_PI*M_PI) / (alpha * alpha);	
	double exp_pref = (2.0 * alpha / sqrt(M_PI));
	double rc2 = rc*rc;
	
	// vectors
    vector<vector<double>> cell = {}; // positions of all particles
	vector<double> q_list = {};	  // charge of all particles
        vector<double> m_list = {};	  // mass of all particles
	// primitive lattice vectors of Na and Cl in NaCl
	vector<vector<double>> Cl_lattice_vec = {{0,0,0}, {0,a/2,a/2}, {a/2,0,a/2}, {a/2,a/2,0}};  
	vector<vector<double>> Na_lattice_vec = {{a/2,0,0}, {0,a/2,0}, {0,0,a/2}, {a/2,a/2,a/2}};
    vector<vector<int>> neighbour_list; // neighbour list for real space sum
	int total;  			            // total number of particles
	
	// MPI and FFTW initialisation variables
    MPI_Comm comm;
    int rank, size;
    ptrdiff_t local_n0;
    ptrdiff_t local_0_start;
    ptrdiff_t alloc_local;
    fftw_complex *dataout, *dataxin, *datayin, *datazin;
    double *datain, *dataxout, *datayout, *datazout;
	fftw_plan plan, planx, plany, planz;
	int N2_half = N2/2 + 1;
	int N2_pad  = 2 * N2_half;

	// constructor
        ewald(MPI_Comm c = MPI_COMM_WORLD)
       	    : comm(c)
	{

            // copy the lattice vectors into a supercell
            for (int i = 0; i < N_cell; i++) {
                for (int j = 0; j < N_cell; j++) {
                    for (int k = 0; k < N_cell; k++) {
                        double ax = a*i;
                        double ay = a*j;
                        double az = a*k;
                        for (const auto &q : Cl_lattice_vec) {
                            cell.push_back({q[0]+ax, q[1]+ay, q[2]+az});
                            q_list.push_back(-charge);
                            m_list.push_back(Cl_mass);
			}
                        for (const auto &q : Na_lattice_vec) {
                            cell.push_back({q[0]+ax, q[1]+ay, q[2]+az});
                            q_list.push_back(charge);
			                m_list.push_back(Na_mass);
                        }
                    }
                }
            } 
	    total = cell.size();

	    // compute neighbour list
	    neighbour_list.assign(total, std::vector<int>());
	    for (int i = 0; i<total; i++) {
	       	for (int j = i+1; j<total; j++) {
        	    double dx = min_image_delta(cell[i][0], cell[j][0], L);
        	    double dy = min_image_delta(cell[i][1], cell[j][1], L);
        	    double dz = min_image_delta(cell[i][2], cell[j][2], L);
           	    double r2 = dx*dx + dy*dy + dz*dz;
		    if (r2 < rc2) {
			    neighbour_list[i].push_back(j);
			    neighbour_list[j].push_back(i);
		        }
		    }
	    }
       
 	// initialise MPI and FFTW
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        fftw_mpi_init();

        // Get local slab size
        alloc_local = fftw_mpi_local_size_3d(
            N0, N1, N2_half, comm,
            &local_n0, &local_0_start
        );

        // Allocate distributed array
	    datain = fftw_alloc_real(2 * alloc_local);
	    assert(datain != nullptr);
        dataout = fftw_alloc_complex(alloc_local);
        assert(dataout != nullptr);

        dataxin = fftw_alloc_complex(alloc_local);
        assert(dataxin != nullptr);
        datayin = fftw_alloc_complex(alloc_local);
        assert(datayin != nullptr);
        datazin = fftw_alloc_complex(alloc_local);
        assert(datazin != nullptr);

        dataxout = fftw_alloc_real(2 * alloc_local);
        assert(dataxout != nullptr);
        datayout = fftw_alloc_real(2 * alloc_local);
        assert(datayout != nullptr);
        datazout = fftw_alloc_real(2 * alloc_local);
        assert(datazout != nullptr);
        // Zero array
        zero();

        // Create plans
        plan = fftw_mpi_plan_dft_r2c_3d(
            N0, N1, N2,
            datain, dataout,
            comm,
            FFTW_MEASURE
        );

        planx = fftw_mpi_plan_dft_c2r_3d(
            N0, N1, N2,
            dataxin, dataxout,
            comm,
            FFTW_MEASURE
        );
        plany = fftw_mpi_plan_dft_c2r_3d(
            N0, N1, N2,
            datayin, datayout,
            comm,
            FFTW_MEASURE
        );
        planz = fftw_mpi_plan_dft_c2r_3d(
            N0, N1, N2,
            datazin, datazout,
            comm,
            FFTW_MEASURE
        );	
    }

    ~ewald() {
        fftw_destroy_plan(plan);
        fftw_destroy_plan(planx);
	    fftw_destroy_plan(plany);
	    fftw_destroy_plan(planz);
        fftw_free(datain);
	    fftw_free(dataout);
        fftw_free(dataxin);
        fftw_free(datayin);
        fftw_free(datazin);
        fftw_free(dataxout);
        fftw_free(datayout);
        fftw_free(datazout);
    }

    // zero the local array
    void zero() {
    // zero real buffers
    for (ptrdiff_t r = 0; r < 2 * alloc_local; ++r) {
        datain[r]   = 0.0;
        dataxout[r] = 0.0;
        datayout[r] = 0.0;
        datazout[r] = 0.0;
    }
    // zero complex buffers (alloc_local fftw_complex)
    for (ptrdiff_t c = 0; c < alloc_local; ++c) {
        dataout[c][0] = dataout[c][1] = 0.0;
        dataxin[c][0] = dataxin[c][1] = 0.0;
        datayin[c][0] = datayin[c][1] = 0.0;
        datazin[c][0] = datazin[c][1] = 0.0;
    }
    }
/*
 * Function: charge_assign
 * ----------------------
 * assigns charges to FFT grid using CIC
 *
 */   
   void charge_assign(vector<vector<double>> pos) {
        zero();
        
	    // CIC charge assignment
        for (int p = 0; p < total; ++p) {
            double x = pos[p][0];
            double y = pos[p][1];
            double z = pos[p][2];
            double q = q_list[p];

            double i0 = (x / ds);
            double j0 = (y / ds);
            double k0 = (z / ds);

            int i1 = floor(i0);
            int j1 = floor(j0);
            int k1 = floor(k0);
            
	    // wrap
            i1 = (i1 + N0) % N0;
            j1 = (j1 + N1) % N1;
            k1 = (k1 + N2) % N2;
            
	        double di = i0 - (double)i1;
            double dj = j0 - (double)j1;
            double dk = k0 - (double)k1;

        // loop over all 8 nearest grid points
	    for (int a = 0; a <= 1; a++) {
                double wx = (a ? di : 1.0 - di);
                for (int b = 0; b <= 1; b++) {
                    double wy = (b ? dj : 1.0 - dj);
                    for (int c = 0; c <= 1; c++) {
                        double wz = (c ? dk : 1.0 - dk);
                        int x0 = i1 + a;
                        int y0 = j1 + b;
                        int z0 = k1 + c;
                        
			            x0 = (x0 % N0 + N0) % N0;
			            y0 = (y0 % N1 + N1) % N1;
			            z0 = (z0 % N2 + N2) % N2;

			            double W = wx * wy * wz;
                        double contrib = q * W;
                        int local_i = x0 - static_cast<int>(local_0_start);
                        // check if rank owns the grid point
                        if (0 <= local_i && local_i < static_cast<int>(local_n0)) {
                            ptrdiff_t ind_local = (static_cast<ptrdiff_t>(local_i) * N1 + y0) * N2_pad + z0;
                            datain[ind_local] += contrib;
                        }
                    }
                }
            }
        }
    }	    
    
    // get charge of ith ion 
    double get_charge(int i) {
        return q_list[i];
    }

    // wrap position into the cell
    inline double wrap_pos_into_box(double x, double L) {
        // Shift into [0, L)
        x = fmod(x, L);
        if (x < 0) x += L;
        return x;
    }
    
    // compute min distance seperating particles for real space sum
    inline double min_image_delta(double xi, double xj, double L) {
        double dx = xi - xj;
        return dx - round(dx / L) * L;
    }

    vector<vector<double>> get_positions() {  
        return cell;
    }
/*
 * Function: PME
 * ----------------------
 * Steps:
 *	1. Call charge_assign function
 *	2. Execute Forward FFT
 *	3. Calculate Fourier coefficients from data and add to x,y,z FFT grids
 *	4. Execute 3 Backward FFTs
 */
    // Compute FFT, multiply by Fourier coefficients, compute IFFT
    void PME(vector<vector<double>> pos) {
        size_t n = total;
        // assign charges to grid
        charge_assign(pos);

        // Perform forward FFT
        fftw_execute(plan);

        // reciprocal factor
        for (ptrdiff_t i = 0; i < local_n0; ++i) {
            ptrdiff_t global_i = i + local_0_start;
            for (int j = 0; j < N1; ++j) {
                for (int k = 0; k < N2_half; ++k) {
                    ptrdiff_t ind_local = (i * N1 + j) * (N2/2 + 1) + k;

                        int i_r = (global_i > N0/2) ? global_i - N0 : global_i;
                        int j_r = (j > N1/2)        ? j - N1 : j;
                        int k_r = k;

                        double Gx, Gy, Gz, G2;
                        Gx = i_r/(L);
                        Gy = j_r/(L);
                        Gz = k_r/(L);
                        G2 = Gx*Gx + Gy*Gy + Gz*Gz;

                        // skip G^2 = 0 term
			            if (fabs(G2) < 1e-12) {
                            dataxin[ind_local][0] = 0;
                            dataxin[ind_local][1] = 0;
                            
                            datayin[ind_local][0] = 0;
                            datayin[ind_local][1] = 0;
                            
                            datazin[ind_local][0] = 0;
                            datazin[ind_local][1] = 0;
                        }

                        else {
			            // Fourier coefficients
                        double K = (pref / G2) * exp(-G2 * pre_alpha);
                        dataxin[ind_local][0] = K * 2*M_PI * Gx * dataout[ind_local][1];
                        dataxin[ind_local][1] = -K * 2*M_PI * Gx * dataout[ind_local][0];

                        datayin[ind_local][0] = K * 2*M_PI * Gy * dataout[ind_local][1];
                        datayin[ind_local][1] = -K * 2*M_PI * Gy * dataout[ind_local][0];

                        datazin[ind_local][0] = K * 2*M_PI * Gz * dataout[ind_local][1];
                        datazin[ind_local][1] = -K * 2*M_PI * Gz * dataout[ind_local][0];			
                        }
                }
            }
        }

        // perform inverse FFTs    
        fftw_execute(planx);
        fftw_execute(plany);
        fftw_execute(planz);	
    }

    // Access local data in FFT grid
    double& at(ptrdiff_t local_i, ptrdiff_t j, ptrdiff_t k, int dim) {
        assert(local_i >= 0 && local_i < local_n0);
        assert(j >= 0 && j < N1);
        assert(k >= 0 && k < N2);
        assert(dim >= 0 && dim < 3);
        ptrdiff_t idx = (local_i * N1 + j) * N2_pad + k;
        if (dim==0) return dataxout[idx];
        else if (dim==1) return datayout[idx];
        else return datazout[idx];
    }

/*
 * Function: get_accel
 * ----------------------
 * Steps:
 *      1. Call PME function
 *      2. Compute real space forces/acceleration (Coulomb and Lennard Jones)
 *      3. Interpolate forces/acceleration from PME grid
 *      4. Communicate and sum all data into single vector stored in all ranks
 */
    vector<vector<double>> get_accel(vector<vector<double>> pos) {
	cell = pos;

    // create a wrapped copy of cell so PME uses positions in [0,L]
    vector<vector<double>> cell_wrapped = cell;
    for (size_t idx = 0; idx < cell_wrapped.size(); ++idx) {
        cell_wrapped[idx][0] = wrap_pos_into_box(cell_wrapped[idx][0], L);
        cell_wrapped[idx][1] = wrap_pos_into_box(cell_wrapped[idx][1], L);
        cell_wrapped[idx][2] = wrap_pos_into_box(cell_wrapped[idx][2], L);
    }
	PME(cell_wrapped);
	size_t n = total;
        
	vector<vector<double>> A_list;
    A_list.assign(n, vector<double>(3, 0.0));
	vector<double> A_flat_local(3 * n, 0.0);
	
	// indexes for splitting real space sum among ranks
	int start = rank * total / size;
	int end = (rank + 1) * total / size;
	
	// real space sum
	for (int i=start; i<end; ++i) {
	    const auto &p_i = cell[i];
        double q_i = q_list[i];
        double m_i = m_list[i];

        for (int j : neighbour_list[i]) {
            if (i==j) continue;
            const auto &p_j = cell[j];

            // wrapped distance between paricles
		    double dx = min_image_delta(p_i[0], p_j[0], L);
            double dy = min_image_delta(p_i[1], p_j[1], L);
            double dz = min_image_delta(p_i[2], p_j[2], L);		
		
		    double r2 = dx*dx + dy*dy + dz*dz;
            if (r2>rc2) continue;
		    if (r2<1e-10) continue;
		    double r = sqrt(r2);

            double q_j = q_list[j];
            double m_j = m_list[j];
            vector<double> unit_r = {dx/r, dy/r, dz/r};
            double Fx = 0;
            double Fy = 0;
            double Fz = 0;

            // Coulomb Interaction
            if (r<rc) {
                double erfc_part = erfc(alpha * r);
                double exp_part = exp_pref * exp(-alpha*alpha*r2);
        	    double F_mag = k * q_i * q_j * (erfc_part / r2 + exp_part / r);

                Fx += F_mag * unit_r[0];
                Fy += F_mag * unit_r[1];
                Fz += F_mag * unit_r[2];
            }

            // Lennard Jones Interaction
            if (r < a) {
                double sigma, epsilon; // sigma in bohr, epsilon in Eh
                if (q_i < 0 && q_j < 0) {
                    sigma = 4.76;
                    epsilon = 6.09e-4;
                }
                else if (q_i > 0 && q_j > 0) {
                    sigma = 7.275;
                    epsilon = 5.52e-5;
                }
                else {
                    sigma = 6.02;
                    epsilon = 1.83e-4;
                }
                double LJ_mag = (48/r) * epsilon *  ( pow(sigma/r, 12) - 0.5*pow(sigma/r,6) );
		            Fx += unit_r[0] * LJ_mag;
                    Fy += unit_r[1] * LJ_mag;
                    Fz += unit_r[2] * LJ_mag;
                }

		    // add all force data to flat array
            A_flat_local[3*i + 0] += Fx/m_i;
            A_flat_local[3*i + 1] += Fy/m_i;
            A_flat_local[3*i + 2] += Fz/m_i;
	    }
	}

    // interpolate reciprocal space force from FFT grid
	for (int i = 0; i < n; ++i) {
        double px = cell_wrapped[i][0]/ds, py = cell_wrapped[i][1]/ds, pz = cell_wrapped[i][2]/ds;
        int ix = floor(px), iy = floor(py), iz = floor(pz);

	    // wrap
	    ix = (ix + N0) % N0;
	    iy = (iy + N1) % N1;
        iz = (iz + N2) % N2;
	   
	    double dx = px - (double)ix;
	    double dy = py - (double)iy;
	    double dz = pz - (double)iz;
	    double px_val = 0.0, py_val = 0.0, pz_val = 0.0;
        for (int a = 0; a <= 1; a++) {
            double wx = (a ? dx : 1.0 - dx);
            for (int b = 0; b <= 1; b++) {
                double wy = (b ? dy : 1.0 - dy);
                for (int c = 0; c <= 1; c++) {
                    double wz = (c ? dz : 1.0 - dz);
                    int x0 = ix + a;
                    int y0 = iy + b;
                    int z0 = iz + c;

                    x0 = (x0 % N0 + N0) % N0;
                    y0 = (y0 % N1 + N1) % N1;
                    z0 = (z0 % N2 + N2) % N2;			
			        double W = wx * wy * wz;
			        int local_ix = x0 - static_cast<int>(local_0_start);
			        if (0 <= local_ix && local_ix < static_cast<int>(local_n0)) {
                	    double &cx = at(local_ix, y0, z0, 0);
                	    double &cy = at(local_ix, y0, z0, 1);
                	    double &cz = at(local_ix, y0, z0, 2);			   
                	    px_val += cx*W;
                	    py_val += cy*W;
                	    pz_val += cz*W;			    
			        }
		        }   
		    }
	    }
        A_flat_local[3*i + 0] += k * (q_list[i] / m_list[i]) * px_val;
        A_flat_local[3*i + 1] += k * (q_list[i] / m_list[i]) * py_val;
        A_flat_local[3*i + 2] += k * (q_list[i] / m_list[i]) * pz_val;
	}	

	// Communicate forces from each rank to all ranks
	vector<double> A_flat_global(3 * n, 0.0);
	MPI_Allreduce(A_flat_local.data(), A_flat_global.data(), 3 * (int)n, MPI_DOUBLE, MPI_SUM, comm);
	for (int i = 0; i < n; ++i) {
	    A_list[i][0] += A_flat_global[3*i + 0];
        A_list[i][1] += A_flat_global[3*i + 1];
        A_list[i][2] += A_flat_global[3*i + 2];
	}
	return A_list;
    }

};



int main(int argc, char **argv) {
    // initialise MPI
    MPI_Init(&argc, &argv); 
    {
    // call class 
    ewald f;

    int N = 500;     			// no. of iterations for simulation
    double dt = 100; 			// time step (unit = 2.42e-17 s)
    double temp = 300; 			// Temperature in Kelvin
    double v = sqrt(temp/5.65e9); 	// velocity magnitude in au

    // vectors for velocity Verlet
    vector<vector<vector<double>>> pos_list;
    vector<vector<double>> pos = f.get_positions();
    pos_list.push_back(pos);
    vector<array<double,3>> v_list;
    vector<double> t_list;
    t_list.push_back(0.0);
    vector<vector<double>> a0;
    vector<vector<double>> a1;
    int size = pos.size();
    double L = f.L;
    v_list.resize(size);
        
    // set initial velocities
    if (f.rank == 0) {
        for (int i = 0; i < size; i++) {
            // spherical angles
            double phi = M_PI * ((double)rand()) / RAND_MAX;
            double theta = 2 * M_PI * ((double)rand()) / RAND_MAX;

            double vx = v * sin(phi) * cos(theta);
            double vy = v * sin(phi) * sin(theta);
            double vz = v * cos(phi);
            v_list[i] = {vx, vy, vz};
        }
    }
   
    // send initial velocities to all ranks
    MPI_Bcast(v_list.data(), 3 * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

/*
 * Velocity Verlet Integrator (VVI) loop
 * ----------------------
 * Steps:
 *      1. Call get_accel with current positions
 *      2. Compute new positions according to first (VVI) eqn
 *      3. write all positions to file
 *      4. Call get_accel with new positions
 *      5. Compute new velocities according to second (VVI) eqn
 *      6. Repeat.
 */    
    std::ofstream traj_file(filename); 
    for (int i = 1; i < N; i++) {
	// write frame no. and time to file
	if (f.rank == 0 && i % 1 == 0) {
            traj_file << size << "\n";
            traj_file << "Frame " << i << ": T=" << t_list[i-1] << "\n";        
	}

	if (i==1) {
	    a0 = f.get_accel(pos);  // compute forces
	}
	else a0 = a1;
	
        for (int j = 0; j < size; j++) {
            // calculate positions according to velocity and force
	        double x = pos[j][0] + v_list[j][0]*dt + 0.5 * a0[j][0] * dt * dt;
            double y = pos[j][1] + v_list[j][1]*dt + 0.5 * a0[j][1] * dt * dt;
            double z = pos[j][2] + v_list[j][2]*dt + 0.5 * a0[j][2] * dt * dt;
            // wrap positions into cell
	        pos[j][0] = f.wrap_pos_into_box(x, L);
            pos[j][1] = f.wrap_pos_into_box(y, L);
            pos[j][2] = f.wrap_pos_into_box(z, L);
            
	    // write positions into file
	    if (f.rank == 0 && i % 1 == 0) {
		if (f.q_list[j] > 0) traj_file << "Na" << " ";
                else traj_file << "Cl" << " ";            
                traj_file << pos[j][0] << " ";
                traj_file << pos[j][1] << " ";
                traj_file << pos[j][2] << "\n";	
	    }
	}

	a1 = f.get_accel(pos);  // compute new forces

	// calculate velocities
        for (int j = 0; j < size; j++) {
            double vx = v_list[j][0] + 0.5 * (a0[j][0] + a1[j][0]) * dt;
            double vy = v_list[j][1] + 0.5 * (a0[j][1] + a1[j][1]) * dt;
            double vz = v_list[j][2] + 0.5 * (a0[j][2] + a1[j][2]) * dt;
            v_list[j] = {vx,vy,vz};
        }
        t_list.push_back(i*dt);
    }  
    }
    
    MPI_Finalize();
    return 0;
}
