#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>
#include <bits/stdc++.h>
#include <fstream>
#include <list>
#include <numeric>
#include <mpi.h>
#include <fftw3-mpi.h>
#include <cmath>
#include <cstdio>
using namespace std;

string filename = "PME_traj_scale.xyz";
string filename2 = "Force_N4_alpha1.dat";
string filename3 = "PME_Energy_N4_T300_a02_FFT10_rc15.dat";


class ewald {
    public:
        // physical parameters
        double alpha = 0.2;
        int N_cell = 4;
		int gspace = 10;
		ptrdiff_t N = 2*N_cell*gspace; 
        //ptrdiff_t N = 256;
		ptrdiff_t N0 = N, N1 = N, N2 = N;
        bool ignorereal = false;
		double a0 = 0.529177;
		double a = 5.64 / a0;
		double L = a*N_cell;
        double rc = 1.5*a;
		double ds = L / N0;
        double V_cell = pow(L,3);
        double k = 1; // Coulomb factor converted to au
        
		// precomputed factors
		double pref = 1.0 / (V_cell*M_PI);
        double pre_alpha = (M_PI*M_PI) / (alpha * alpha);	
		double exp_pref = (2.0 * alpha / sqrt(M_PI));
		double rc2 = rc*rc;
		double energy = 0;
		double energy_local = 0;
		double rec_energy, rec_local = 0;
		double LJ_energy, LJ_local = 0;
		double real_energy, real_local = 0;
		double self_energy, self_local = 0;

	// vectors
        vector<vector<double>> cell = {};
		vector<double> q_list = {};
        vector<double> m_list = {};
		int total;
        int r_i;
	
		vector<vector<int>> neighbour_list;

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

        ewald(MPI_Comm c = MPI_COMM_WORLD)
       	    : comm(c)
	{

            // copy the lattice vectors into a supercell
	    	double charge = 1;
			double mass = 1;
			double N_grid = 2 * N_cell;
			for (int i = 0; i < N_grid; ++i) {
    			    for (int j = 0; j < N_grid; ++j) {
        			for (int k = 0; k < N_grid; ++k) {

            			    const double ax = 0.5 * a * i;
            			    const double ay = 0.5 * a * j;
            		  	    const double az = 0.5 * a * k;

				    if ( (i + j + k) % 2 == 0) {
				        charge = -.885;
					mass = 65092;
					//mass = 35.5e-2;
				    }
				    else {
					charge = .885;
					mass = 42213;
					//mass = 23.0e-2;
				    }
                		    cell.push_back({ax, ay, az});
                	       	    q_list.push_back(charge);
                		    m_list.push_back(mass);
            			}
           		    }
        		}

	    total = cell.size();
	    neighbour_list.assign(total, std::vector<int>());
	    // compute neighbour list
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


            // choose ion in middle of supercell
            int i0 = N_cell, j0 = N_cell, k0 = N_cell;
            size_t cellIndex = 4*i0*N_cell*N_cell + 2*j0*N_cell + k0;
	    	r_i = cellIndex*1; // multiply by 8 as there are 8 lattice vectors
        
 
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




   // Zero the local array
    void zero() {
    // zero real buffers (2*alloc_local doubles)
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


   void charge_assign(vector<vector<double>> pos) {
        zero();
	//cell = pos;
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
			if (0 <= local_i && local_i < static_cast<int>(local_n0)) {
                            ptrdiff_t ind_local = (static_cast<ptrdiff_t>(local_i) * N1 + y0) * N2_pad + z0;
                            datain[ind_local] += contrib;
                        }
                    }
                }
            }
        }
    }	    
    

    vector<double> get_pos() {
        return cell[r_i];
    }
    void shift_pos(double x) {
        cell[r_i][0] += x;
    }

    double get_charge(int i) {
        return q_list[i];
    }

    int get_middle_idx() {
        return r_i;
    }

    inline double wrap_pos_into_box(double x, double L) {
        // Shift into [0, L)
        x = fmod(x, L);
        if (x < 0) x += L;
        return x;
    }
    inline double min_image_delta(double xi, double xj, double L) {
        double dx = xi - xj;
        return dx - round(dx / L) * L;
    }
    vector<vector<double>> get_positions() {  
        return cell;
    }

    void PME(vector<vector<double>> pos) {
        size_t n = total;
        charge_assign(pos);

        // Perform FFT
        fftw_execute(plan);
		rec_local = 0;
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

                        if (fabs(G2) < 1e-12) {
                            dataxin[ind_local][0] = 0;
                            dataxin[ind_local][1] = 0;
                            
                            datayin[ind_local][0] = 0;
                            datayin[ind_local][1] = 0;
                            
                            datazin[ind_local][0] = 0;
                            datazin[ind_local][1] = 0;
                        }

                        else {
                        double K = (pref / G2) * exp(-G2 * pre_alpha);
                        dataxin[ind_local][0] = K * 2*M_PI * Gx * dataout[ind_local][1];
                        dataxin[ind_local][1] = -K * 2*M_PI * Gx * dataout[ind_local][0];

                        datayin[ind_local][0] = K * 2*M_PI * Gy * dataout[ind_local][1];
                        datayin[ind_local][1] = -K * 2*M_PI * Gy * dataout[ind_local][0];

                        datazin[ind_local][0] = K * 2*M_PI * Gz * dataout[ind_local][1];
                        datazin[ind_local][1] = -K * 2*M_PI * Gz * dataout[ind_local][0];			
                        
		    double rho2 = dataout[ind_local][0]*dataout[ind_local][0] + dataout[ind_local][1]*dataout[ind_local][1];
			double mult = (k == 0 || k == N2_half - 1) ? 0.5 : 1.0;
			rec_local += mult * K * rho2;
			
			}
                }
            }
        }
	//energy_local += 0.5 * rec_local;
        // perform inverse FFTs    
        fftw_execute(planx);
        fftw_execute(plany);
        fftw_execute(planz);	
    }

    // Access local data
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

    vector<vector<double>> get_force(vector<vector<double>> pos) {
	cell = pos;
        // create a wrapped copy of cell so PME uses positions in [0,L]
        vector<vector<double>> cell_wrapped = cell;
        for (size_t idx = 0; idx < cell_wrapped.size(); ++idx) {
            cell_wrapped[idx][0] = wrap_pos_into_box(cell_wrapped[idx][0], L);
            cell_wrapped[idx][1] = wrap_pos_into_box(cell_wrapped[idx][1], L);
            cell_wrapped[idx][2] = wrap_pos_into_box(cell_wrapped[idx][2], L);
        }

	// all energy contributions
	energy = 0.0;
	energy_local = 0.0;
	rec_local = 0.0;
	rec_energy = 0.0;
	real_local = 0.0;
	real_energy = 0.0;
	LJ_local = 0.0;
	LJ_energy = 0.0;
	self_energy = 0.0;
	
	PME(cell_wrapped);
	size_t n = total;
    vector<vector<double>> F_list;
    F_list.assign(n, vector<double>(3, 0.0));
	vector<double> F_flat_local(3 * n, 0.0);
	int start = rank * total / size;
	int end = (rank + 1) * total / size;
		
	// real space sum
	for (int i=start; i<end; ++i) {
	    const auto &p_i = cell[i];
        double q_i = q_list[i];
        double m_i = m_list[i];

        for (int j : neighbour_list[i]) {
	        if (ignorereal) continue;
                if (i==j) continue;
                const auto &p_j = cell[j];

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
		    
		    double pair_real = 0.5 * q_i * q_j * erfc(alpha * r) / r;
		    real_local += pair_real;
		    //energy_local += pair_real;
		}

                // Lennard Jones Interaction
                if (r < 1*a) {
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
                    //double LJ_mag = (48/r) * epsilon * ( sr12 - 0.5 * sr6);
		    
		    double pair_LJ = 4.0 * epsilon * (pow(sigma/r, 12) - pow(sigma/r, 6));
		    LJ_local += 0.5 * pair_LJ;
		    //energy_local += pair_LJ;		   

		    Fx += unit_r[0] * LJ_mag;
            Fy += unit_r[1] * LJ_mag;
            Fz += unit_r[2] * LJ_mag;
                }

        F_flat_local[3*i + 0] += Fx/m_i;
		F_flat_local[3*i + 1] += Fy/m_i;
		F_flat_local[3*i + 2] += Fz/m_i;
	    }
	}

        // reciprocal space part
	for (int i = 0; i < n; ++i) {
    	double px = cell_wrapped[i][0]/ds, py = cell_wrapped[i][1]/ds, pz = cell_wrapped[i][2]/ds;
	    //int ix = round(px/ds), iy = round(py/ds), iz = round(pz/ds);
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
            F_flat_local[3*i + 0] += k * (q_list[i] / m_list[i]) * px_val;
            F_flat_local[3*i + 1] += k * (q_list[i] / m_list[i]) * py_val;
            F_flat_local[3*i + 2] += k * (q_list[i] / m_list[i]) * pz_val;
	}	

	vector<double> F_flat_global(3 * n, 0.0);
	MPI_Allreduce(F_flat_local.data(), F_flat_global.data(), 3 * (int)n, MPI_DOUBLE, MPI_SUM, comm);
	for (int i = 0; i < n; ++i) {
	    F_list[i][0] += F_flat_global[3*i + 0];
        F_list[i][1] += F_flat_global[3*i + 1];
        F_list[i][2] += F_flat_global[3*i + 2];
	}
	        
        MPI_Allreduce(&real_local, &real_energy, 1, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(&rec_local, &rec_energy, 1, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(&LJ_local, &LJ_energy, 1, MPI_DOUBLE, MPI_SUM, comm);
        self_energy = 0.0;
        for (size_t ii = 0; ii < q_list.size(); ++ii) {
            self_energy -= q_list[ii] * q_list[ii] * alpha / sqrt(M_PI);
        }
	energy = real_energy + rec_energy + LJ_energy + self_energy;

	return F_list;
    }

};



int main(int argc, char **argv) {
    MPI_Init(&argc, &argv); 
    {
    ewald f;
    
    int N = 100000;
    double dt = 20; // unit = 2.42e-17 s
    double temp = 300; // Temperature in Kelvin
    double v = sqrt(temp/5.65e9); // velocity magnitude in au

    vector<vector<vector<double>>> pos_list;
    //f.shift_pos(1);
    vector<vector<double>> pos = f.get_positions();
    pos_list.push_back(pos);

    vector<array<double,3>> v_list;
    vector<double> t_list;
    vector<double> energy_list;
    vector<double> rec_energy;
    vector<double> real_energy;
    vector<double> LJ_energy;
    vector<double> self_energy;
    vector<double> kinetic_energy;
    t_list.push_back(0.0);

    vector<vector<double>> a0;
    vector<vector<double>> a1;
    
    int size = pos.size();
    double L = f.L;
    int m = f.get_middle_idx();
    mt19937 rng( random_device{}() );         // Mersenne Twister engine
    normal_distribution<double> dist(0.0, 0.1);    // Gaussian Distribution
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
            //double vx = 0;
            //double vy = 0;
            //double vz = 0;
            v_list[i] = {vx, vy, vz};

            //pos[i][0] += dist(rng);
            //pos[i][1] += dist(rng);
            //pos[i][2] += dist(rng);
        }
    }
    
    MPI_Bcast(v_list.data(), 3 * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    //vector<vector<double>> p_list;
    //p_list.assign(size, vector<double>(3, 0.0));
    //std::ofstream force_file(filename2);
    
    //std::ofstream traj_file(filename); 
    for (int i = 1; i < N; i++) {
/*	if (f.rank == 0 && i % 1 == 0) {
            traj_file << size << "\n";
            traj_file << "Frame " << i << ": T=" << t_list[i-1] << "\n";        
	}*/

	if (i==1) {
	    //a0 = f.get_force(pos_list[i-1]);
	    a0 = f.get_force(pos);
	}
	else a0 = a1;
        //pos_list.push_back(vector<vector<double>>(size, vector<double>(3)));
	
        for (int j = 0; j < size; j++) {
            //int jq = j / (8*f.N_cell*f.N_cell);
            //ptrdiff_t local_i = jq - f.local_0_start;
            //if (local_i < 0 || local_i >= f.local_n0) continue;
            double x = pos[j][0] + v_list[j][0]*dt + 0.5 * a0[j][0] * dt * dt;
            double y = pos[j][1] + v_list[j][1]*dt + 0.5 * a0[j][1] * dt * dt;
            double z = pos[j][2] + v_list[j][2]*dt + 0.5 * a0[j][2] * dt * dt;
            pos[j][0] = f.wrap_pos_into_box(x, L);
            pos[j][1] = f.wrap_pos_into_box(y, L);
            pos[j][2] = f.wrap_pos_into_box(z, L);
            
/*	    if (f.rank == 0 && i % 1 == 0) {
		if (f.q_list[j] > 0) traj_file << "Na" << " ";
                else traj_file << "Cl" << " ";            
                traj_file << pos[j][0] << " ";
                traj_file << pos[j][1] << " ";
                traj_file << pos[j][2] << "\n";	
	    }*/
	} 

	a1 = f.get_force(pos);
	double KE = 0;
        for (int j = 0; j < size; j++) {
            double vx = v_list[j][0] + 0.5 * (a0[j][0] + a1[j][0]) * dt;
            double vy = v_list[j][1] + 0.5 * (a0[j][1] + a1[j][1]) * dt;
            double vz = v_list[j][2] + 0.5 * (a0[j][2] + a1[j][2]) * dt;
            v_list[j] = {vx,vy,vz};
	    double v_sq = vx*vx + vy*vy + vz*vz;
            KE += 0.5 * f.m_list[j] * v_sq;
	}
        t_list.push_back(i*dt*2.42e-5);
        energy_list.push_back(f.energy+KE);
        kinetic_energy.push_back(KE);
	//rec_energy.push_back(f.rec_energy);
        //real_energy.push_back(f.real_energy);
        //LJ_energy.push_back(f.LJ_energy);
        //self_energy.push_back(f.self_energy);
    
    }  
	int equil = 1000;
    if (f.rank==0) {
	std::ofstream energy_file(filename3);
	for (int i = equil; i < N-1; i++) {
	    double delta_E = fabs(energy_list[i] - energy_list[equil]) / fabs(energy_list[equil]);
	    energy_file << t_list[i] << " ";
	    energy_file << energy_list[i] << " ";
	    energy_file << kinetic_energy[i] << " ";
	    energy_file << rec_energy[i] << " ";
	    energy_file << real_energy[i] << " ";
	    energy_file << LJ_energy[i] << " ";
	    energy_file << self_energy[i] << " ";
	    energy_file << delta_E << "\n"; 
	}
    }

/*    if (f.rank==0) {
    std::ofstream traj_file(filename);
    for (int i = 0; i < N; i++) {
        traj_file << size << "\n";
        traj_file << "Frame " << i << ": T=" << t_list[i] << "\n";
        for (int j = 0; j < size; j++) {
            if (f.q_list[j] > 0) traj_file << "Na" << " ";
            else traj_file << "Cl" << " ";
            traj_file << pos_list[i][j][0] << " ";
            traj_file << pos_list[i][j][1] << " ";
            traj_file << pos_list[i][j][2] << "\n";
            //traj_file << t_list[i] << "\n";
        }
        //traj_file << "\n";
    }
    } */
    }
    MPI_Finalize();
    return 0;
}
