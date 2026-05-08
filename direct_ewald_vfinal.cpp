#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>
#include <bits/stdc++.h>
#include <fstream>
#include <list>
#include <numeric>
#include <cmath>
using namespace std;

string filename = "NaCl_traj_N4_T300.xyz";
string filename2 = "Force_N1_anim_fast_005.dat";
class ewald {
    public:
        double a0 = 0.529177;
	double a = 5.64 / a0; // lattice constant in bohr
        int N = 4; // size of supercell in terms of unit cell
        int G_cutoff = 2; // dictates number of terms in reciprocal space sum
        double alpha = 0.14;
        double rc = 1*a; // cutoff radius for real space
        double k = 1;
        bool ignorereal = false;
        vector<double> q_list;
        vector<double> m_list;
        vector<vector<double>> cell = {};

        // initialise lattice vectors for Cl and Na seperately
        vector<vector<double>> Cl_lattice_vec = {{0,0,0}, {0,a/2,a/2}, {a/2,0,a/2}, {a/2,a/2,0}};
        //Na_lattice_vec = {{a/2,0,0}, {a/2,a/2,a/2}, {a,0,a/2}, {a,a/2,0}};
        vector<vector<double>> Na_lattice_vec = {{a/2,0,0}, {0,a/2,0}, {0,0,a/2}, {a/2,a/2,a/2}};

        // generate reciprocal lattice vectors
        double L = a * (double)N;
        double Vcell = pow(L, 3);
        double rf = 2*M_PI/(L);
        vector<vector<double>> rec_lattice_vec = {{rf,0,0}, {0,rf,0}, {0,0,rf}};
        //vector<vector<double>> rec_lattice_vec = {{-rf,rf,rf}, {rf,-rf,rf}, {rf,rf,-rf}};
        //rec_lattice_vec = {{-rf,2*rf,2*rf}, {rf,-2*rf,rf}, {rf,rf,-2*rf}};

        size_t total = 0;   // number of ions in real-space cell
        size_t r_i = 0;     // index of chosen ion in 'cell'

        double exp_pref = (2.0 * alpha / sqrt(M_PI));
        double rc2 = rc*rc;

        vector<double> F = {0, 0, 0};

        vector<vector<double>> positions;
        vector<double> check;
        vector<vector<double>> G_list;
        vector<double> G_pref;  
        vector<double> G2_list;  
        vector<vector<int>> neighbour_list;


        ewald() {
            // copy the lattice vectors into a supercell
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    for (int k = 0; k < N; k++) {
                        double ax = a*i;
                        double ay = a*j;
                        double az = a*k;
                        for (const auto &q : Cl_lattice_vec) {
                            cell.push_back({q[0]+ax, q[1]+ay, q[2]+az});
                            q_list.push_back(-.885);    
                            m_list.push_back(65092);
                            // m_list.push_back(1.0);
                        }
                        for (const auto &q : Na_lattice_vec) {
                            cell.push_back({q[0]+ax, q[1]+ay, q[2]+az});
                            q_list.push_back(.885);    
                            m_list.push_back(42213);
                            // m_list.push_back(1.0);

                        }
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
            int i0 = N/2, j0 = N/2, k0 = N/2;
            //size_t cellIndex = 4*N*N*N + 6*N*N + 3*N; // since there are N^2 j,k pairs for each i and N k pairs for each j
            //size_t cellIndex = 4*N*N*N + 2*N*N + N;
            size_t cellIndex = 4*i0*N*N + 2*j0*N + k0;
            //r_i = cellIndex*8; // multiply by 8 as there are 8 lattice vectors
            //printf("chosen ion is [%8.3f,%8.3f,%8.3f] \n", cell[r_i][0]/a,cell[r_i][1]/a,cell[r_i][2]/a);

            for (int h = -G_cutoff; h <= G_cutoff; ++h) {
                for (int j = -G_cutoff; j <= G_cutoff; ++j) {
                    for (int k = -G_cutoff; k <= G_cutoff; ++k) {
                        if (h==0 && j==0 && k==0) continue;
                        double Gx = h*rec_lattice_vec[0][0] + j*rec_lattice_vec[1][0] + k*rec_lattice_vec[2][0];
                        double Gy = h*rec_lattice_vec[0][1] + j*rec_lattice_vec[1][1] + k*rec_lattice_vec[2][1];
                        double Gz = h*rec_lattice_vec[0][2] + j*rec_lattice_vec[1][2] + k*rec_lattice_vec[2][2];
                        double G2 = Gx*Gx + Gy*Gy + Gz*Gz;
                        //if (G2 < 1e-12) continue;
                        G_list.push_back({Gx, Gy, Gz});
                        G2_list.push_back(G2);
                        double pref = (4.0*M_PI / Vcell) * exp(-G2/(4.0*alpha*alpha)) / G2;
                        G_pref.push_back(pref);
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
        // Shift into [-L/2, L/2)
        x = fmod(x + 0.5*L, L);
        if (x < 0) x += L;
        return x - 0.5*L;
    }

    inline double min_image_delta(double xi, double xj, double L) {
        double dx = xi - xj;
        return dx - round(dx / L) * L;
    }
    vector<vector<double>> get_positions() {  
        return cell;
    }

    vector<vector<double>> get_force(vector<vector<double>> pos) {
        cell = pos;
        size_t n = total;
        vector<vector<double>> F_list;
        F_list.assign(n, vector<double>(3, 0.0));

        // create a wrapped copy of cell so G.r uses positions in [-L/2,L/2]
        std::vector<std::vector<double>> cell_wrapped = cell;
        for (size_t idx = 0; idx < cell_wrapped.size(); ++idx) {
            cell_wrapped[idx][0] = wrap_pos_into_box(cell_wrapped[idx][0], L);
            cell_wrapped[idx][1] = wrap_pos_into_box(cell_wrapped[idx][1], L);
            cell_wrapped[idx][2] = wrap_pos_into_box(cell_wrapped[idx][2], L);
        }

        // Compute Sreal/Simag for each G
        size_t NG = G_list.size();
        vector<double> Sreal_list(NG, 0.0), Simag_list(NG, 0.0);
        for (size_t ig = 0; ig < NG; ++ig) {
            const auto &G = G_list[ig];
            double Sreal = 0.0, Simag = 0.0;
            for (size_t j = 0; j < n; ++j) {
                const auto &rj = cell_wrapped[j];
                // const auto &rj = cell[j];
                double G_dot_rj = G[0]*rj[0] + G[1]*rj[1] + G[2]*rj[2];
                Sreal += q_list[j] * cos(G_dot_rj);
                Simag += q_list[j] * sin(G_dot_rj);
            }
            Sreal_list[ig] = Sreal;
            Simag_list[ig] = Simag;
        }

        for (int i=0; i<n; ++i) {
            const auto &p_i = cell[i];
            double q_i = q_list[i];
            double m_i = m_list[i];

            // real space sum
            //for (int j=i+1; j<n; ++j) {
            for (int j : neighbour_list[i]) {
                if (ignorereal) continue;
                const auto &p_j = cell[j];

                // double dx = wrap_pos_into_box(p_i[0] - p_j[0], L);
                // double dy = wrap_pos_into_box(p_i[1] - p_j[1], L);
                // double dz = wrap_pos_into_box(p_i[2] - p_j[2], L);
                double dx = min_image_delta(p_i[0], p_j[0], L);
                double dy = min_image_delta(p_i[1], p_j[1], L);
                double dz = min_image_delta(p_i[2], p_j[2], L);
                double r2 = dx*dx + dy*dy + dz*dz;
                if (r2 > rc2) continue;
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

                F_list[i][0] += Fx/m_i;
                F_list[i][1] += Fy/m_i;
                F_list[i][2] += Fz/m_i;

                F_list[j][0] += -Fx/m_j;
                F_list[j][1] += -Fy/m_j;
                F_list[j][2] += -Fz/m_j;
            }

        // reciprocal space sum
        for (int ig = 0; ig < NG; ++ig) {
            const auto &G = G_list[ig];
            double G_dot_ri = G[0]*p_i[0] + G[1]*p_i[1] + G[2]*p_i[2];
            double Sreal = Sreal_list[ig];
            double Simag = Simag_list[ig];
            double phase = q_i * (Sreal * sin(G_dot_ri) - Simag * cos(G_dot_ri));
            double pref = k * G_pref[ig] * phase / m_i;
                            
            F_list[i][0] += G[0] * pref;
            F_list[i][1] += G[1] * pref;
            F_list[i][2] += G[2] * pref;
        }
                        
        }
        return F_list;
    }

};

int main() {
    ewald f;

    int N = 500;
    double dt = 200;
    double temp = 300; // Temperature in Kelvin
    double v = sqrt(temp/5.65e9); // velocity magnitude in au    
    vector<vector<vector<double>>> pos_list;
    f.shift_pos(0);
    vector<vector<double>> pos = f.get_positions();
    pos_list.push_back(pos);

    vector<vector<double>> v_list;
    vector<double> t_list;
    t_list.push_back(0.0);

    vector<vector<double>> a0;
    vector<vector<double>> a1;
    
    int size = pos.size();
    double L = f.L;
    int m = f.get_middle_idx();


    mt19937 rng( random_device{}() );         // Mersenne Twister engine
    normal_distribution<double> dist(0.0, .1);    // Gaussian Distribution
    // set initial velocities
    for (int i = 0; i < size; i++) {

            // spherical angles
            double phi = M_PI * ((double)rand()) / RAND_MAX;
            double theta = 2 * M_PI * ((double)rand()) / RAND_MAX;

            double vx = v * sin(phi) * cos(theta);
            double vy = v * sin(phi) * sin(theta);
            double vz = v * cos(phi);
        v_list.push_back({vx, vy, vz});
        // pos_list[0][i][0] += dist(rng);
        // pos_list[0][i][1] += dist(rng);
        // pos_list[0][i][2] += dist(rng);
    }

    std::ofstream force_file(filename2);
    std::ofstream traj_file(filename);
     for (int i = 1; i < N; i++) {
        traj_file << size << "\n";
        traj_file << "Frame " << i << ": T=" << t_list[i-1] << "\n";

        if (i==1) a0 = f.get_force(pos);
        else a0 = a1;
        pos_list.push_back(vector<vector<double>>(size, vector<double>(3)));

        for (int j = 0; j < size; j++) {
            double x = pos[j][0] + v_list[j][0]*dt + 0.5 * a0[j][0] * dt * dt;
            double y = pos[j][1] + v_list[j][1]*dt + 0.5 * a0[j][1] * dt * dt;
            double z = pos[j][2] + v_list[j][2]*dt + 0.5 * a0[j][2] * dt * dt;
            // pos_list[i][j][0] = f.wrap_pos_into_box(x, L);
            // pos_list[i][j][1] = f.wrap_pos_into_box(y, L);
            // pos_list[i][j][2] = f.wrap_pos_into_box(z, L);
            pos[j][0] = x;
            pos[j][1] = y;
            pos[j][2] = z;

            if (f.q_list[j] > 0) traj_file << "Na" << " ";
            else traj_file << "Cl" << " ";
            traj_file << pos[j][0] << " ";
            traj_file << pos[j][1] << " ";
            traj_file << pos[j][2] << "\n";
        }
        // force_file << a0[m][0] << " ";
        // force_file << a0[m][1] << " ";
        // force_file << a0[m][2] << " ";
        // force_file << (i-1)*dt << "\n";
        a1 = f.get_force(pos);

        for (int j = 0; j < size; j++) {
            double vx = v_list[j][0] + 0.5 * (a0[j][0] + a1[j][0]) * dt;
            double vy = v_list[j][1] + 0.5 * (a0[j][1] + a1[j][1]) * dt;
            double vz = v_list[j][2] + 0.5 * (a0[j][2] + a1[j][2]) * dt;
            v_list[j] = {vx,vy,vz};
        }
        t_list.push_back(i*dt);
    }  

/*     std::ofstream traj_file(filename);
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
    } */

    return 0;
}
