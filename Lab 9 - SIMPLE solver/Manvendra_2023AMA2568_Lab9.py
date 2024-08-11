
#==================================================================================#
#                                    APL 720                                       #
#                      COMPUTATIONAL FLUID DYNAMICS LABORATORY                     #
#                                  ASSIGNMENT 9                                    #
#                      Submitted by: Manvendra Singh Rajawat                       #
#                           Entry Number: 2023AMA2568                              #
#==================================================================================#

'''
==================================================================================
                                LIBRARIES
==================================================================================
'''

import numpy as np
import matplotlib.pyplot as plt

'''
==================================================================================
                                CLASSES
==================================================================================
'''

#=========================== u-velocity nodes ====================================

class u_vel_node():
    def __init__(self, node_num, u_prev, v_prev, p_prev):

        self.N_wall = False
        self.S_wall = False
        self.E_wall = False
        self.W_wall = False

        self.is_boundary(node_num)
        self.is_corner(node_num)

        if self.W_wall:
            self.Fw = RHO*(U_0 + u_prev[node_num - idx_shft]) / 2.0
            self.a_iminus1_J = MU*hy/hx + max(self.Fw, 0)*hy
        else:
            self.Fw = RHO*(u_prev[node_num - 1 - idx_shft] + u_prev[node_num - idx_shft]) / 2.0
            self.a_iminus1_J = MU*hy/hx + max(self.Fw, 0)*hy

        if self.E_wall:
            self.Fe = RHO*(u_prev[node_num - idx_shft] + u_prev[node_num - idx_shft]) / 2.0
            self.a_iplus1_J = 0 + max(-self.Fe, 0)*hy
        else:
            self.Fe = RHO*(u_prev[node_num + 1 - idx_shft] + u_prev[node_num - idx_shft]) / 2.0
            self.a_iplus1_J = MU*hy/hx + max(-self.Fe, 0)*hy

        south_v_node_num = int(((node_num - EPS) // (Nx-1)) + node_num - (Nx - 1))
        north_v_node_num = south_v_node_num + Nx

        # print(node_num, south_v_node_num, north_v_node_num)

        if self.S_wall:
            self.Fs = RHO*(v_S + v_S) / 2.0
            self.a_i_Jminus1 = 2*MU*hx/hy + max(self.Fs, 0)*hx
        else:
            self.Fs = RHO*(v_prev[south_v_node_num - 1 - idx_shft] + v_prev[south_v_node_num - idx_shft]) / 2.0
            self.a_i_Jminus1 = MU*hx/hy + max(self.Fs, 0)*hx

        if self.N_wall:
            self.Fn = RHO*(v_N + v_N) / 2.0
            self.a_i_Jplus1 = 2*MU*hx/hy + max(-self.Fn, 0)*hx
        else:
            self.Fn = RHO*(v_prev[north_v_node_num - 1 - idx_shft] + v_prev[north_v_node_num - idx_shft]) / 2.0
            self.a_i_Jplus1 = MU*hx/hy + max(-self.Fn, 0)*hx

        if self.E_wall:
            self.S = -(p_E - p_prev[node_num - idx_shft])*hy
        else:
            self.S = -(p_prev[node_num + 1 - idx_shft] - p_prev[node_num - idx_shft])*hy

        self.a_i_J = self.a_iminus1_J + self.a_iplus1_J + self.a_i_Jplus1 + self.a_i_Jminus1 - self.Fs*hx - self.Fw*hy + self.Fe*hy + self.Fn*hx

        if self.is_corner(node_num):
            self.updateCoeff_corner()

        elif self.is_boundary(node_num):
            self.updateCoeff_boundary()

    def is_boundary(self, node_num):

        if node_num > 1 and node_num < Nx - 1:
            self.S_wall = True
            return True
        elif node_num in [j*(Nx-1) + 1 for j in range(1, Ny-1)]:
            self.W_wall = True
            return True
        elif node_num in [(j+1)*(Nx-1) for j in range(1, Ny-1)]:
            self.E_wall = True
            return True
        elif node_num > (Nx-1)*Ny - (Nx-2) and node_num < (Nx-1)*Ny:
            self.N_wall = True
            return True

    def is_corner(self, node_num):

        if node_num == 1:
            self.S_wall = True
            self.W_wall = True
            return True
        elif node_num == Nx - 1:
            self.S_wall = True
            self.E_wall = True
            return True
        elif node_num == (Nx-1)*Ny - (Nx-2):
            self.N_wall = True
            self.W_wall = True
            return True
        elif node_num == (Nx-1)*Ny:
            self.N_wall = True
            self.E_wall = True
            return True

    def updateCoeff_boundary(self):

        if self.W_wall:
            self.S += self.a_iminus1_J*U_0
            self.a_iminus1_J = 0

        elif self.S_wall:
            self.S += self.a_i_Jminus1*u_S
            self.a_i_Jminus1 = 0

        elif self.N_wall:
            self.S += self.a_i_Jplus1*u_N
            self.a_i_Jplus1 = 0

        elif self.E_wall:
            self.S += MU*grad_ux_E*hy

    def updateCoeff_corner(self):

        if self.S_wall and self.W_wall:
            self.S += self.a_iminus1_J*U_0 + self.a_i_Jminus1*u_S
            self.a_i_Jminus1 = 0
            self.a_iminus1_J = 0

        elif self.N_wall and self.W_wall:
            self.S += self.a_iminus1_J*U_0 + self.a_i_Jplus1*u_N
            self.a_i_Jplus1 = 0
            self.a_iminus1_J = 0

        elif self.S_wall and self.E_wall:
            self.S += MU*grad_ux_E*hy + self.a_i_Jminus1*u_S
            self.a_i_Jminus1 = 0

        elif self.N_wall and self.E_wall:
            self.S += MU*grad_ux_E*hy + self.a_i_Jplus1*u_N
            self.a_i_Jplus1 = 0

#=========================== v-velocity nodes ====================================

class v_vel_node():
    def __init__(self, node_num, u_prev, v_prev, p_prev):

        self.N_wall = False
        self.S_wall = False
        self.E_wall = False
        self.W_wall = False

        self.is_boundary(node_num)
        self.is_corner(node_num)

        if self.N_wall:
            self.Fn = RHO*(v_N + v_prev[node_num - idx_shft]) / 2.0
            self.a_I_jplus1 = MU*hx/hy + max(-self.Fn, 0)*hx
        else:
            self.Fn = RHO*(v_prev[node_num + Nx - idx_shft] + v_prev[node_num - idx_shft]) / 2.0
            self.a_I_jplus1 = MU*hx/hy + max(-self.Fn, 0)*hx

        if self.S_wall:
            self.Fs = RHO*(v_S + v_prev[node_num - idx_shft]) / 2.0
            self.a_I_jminus1 = MU*hx/hy + max(self.Fs, 0)*hx
        else:
            self.Fs = RHO*(v_prev[node_num - Nx - idx_shft] + v_prev[node_num - idx_shft]) / 2.0
            self.a_I_jminus1 = MU*hx/hy + max(self.Fs, 0)*hx

        south_u_node_num = node_num - int((node_num - EPS) // Nx)
        north_u_node_num = south_u_node_num + (Nx-1)

        # print(node_num, south_u_node_num, north_u_node_num)

        if self.E_wall:
            self.Fe = RHO*(u_prev[north_u_node_num-1 - idx_shft] + u_prev[south_u_node_num-1 - idx_shft]) / 2.0
            self.a_Iplus1_j = 0 + max(-self.Fe, 0)*hy
        else:
            self.Fe = RHO*(u_prev[north_u_node_num - idx_shft] + u_prev[south_u_node_num - idx_shft]) / 2.0
            self.a_Iplus1_j = MU*hy/hx + max(-self.Fe, 0)*hy

        if self.W_wall:
            self.Fw = RHO*(U_0 + U_0) / 2.0
            self.a_Iminus1_j = 2*MU*hy/hx + max(self.Fw, 0)*hy
        else:
            self.Fw = RHO*(u_prev[north_u_node_num - 1 - idx_shft] + u_prev[south_u_node_num - 1 - idx_shft]) / 2.0
            self.a_Iminus1_j = MU*hy/hx + max(self.Fw, 0)*hy

        if self.E_wall:
            self.S = 0
        else:
            self.S = -(p_prev[node_num - int((node_num - EPS) // Nx) + (Nx-1) - idx_shft] - p_prev[node_num - int((node_num-EPS) // Nx) - idx_shft])*hx

        self.a_I_j = self.a_Iminus1_j + self.a_Iplus1_j + self.a_I_jplus1 + self.a_I_jminus1 - self.Fs*hx - self.Fw*hy + self.Fe*hy + self.Fn*hx

        if self.is_corner(node_num):
            self.updateCoeff_corner()

        elif self.is_boundary(node_num):
            self.updateCoeff_boundary()

    def is_boundary(self, node_num):

        if node_num > 1 and node_num < Nx:
            self.S_wall = True
            return True
        elif node_num in [j*Nx + 1 for j in range(1, Ny-2)]:
            self.W_wall = True
            return True
        elif node_num in [j*Nx + Nx for j in range(1, Ny-2)]:
            self.E_wall = True
            return True
        elif node_num > Nx*(Ny-1) - (Nx-1) and node_num < Nx*(Ny-1):
            self.N_wall = True
            return True

    def is_corner(self, node_num):

        if node_num == 1:
            self.S_wall = True
            self.W_wall = True
            return True
        elif node_num == Nx:
            self.S_wall = True
            self.E_wall = True
            return True
        elif node_num == Nx*(Ny-1) - (Nx-1):
            self.N_wall = True
            self.W_wall = True
            return True
        elif node_num == Nx*(Ny-1):
            self.N_wall = True
            self.E_wall = True
            return True

    def updateCoeff_boundary(self):

        if self.W_wall:
            self.S += self.a_Iminus1_j*v_W
            self.a_Iminus1_j = 0

        elif self.S_wall:
            self.S += self.a_I_jminus1*v_S
            self.a_I_jminus1 = 0

        elif self.N_wall:
            self.S += self.a_I_jplus1*v_N
            self.a_I_jplus1 = 0

        elif self.E_wall:
            self.S += MU*grad_vx_E*hy

    def updateCoeff_corner(self):

        if self.S_wall and self.W_wall:
            self.S += self.a_Iminus1_j*v_W + self.a_I_jminus1*v_S
            self.a_I_jminus1 = 0
            self.a_Iminus1_j = 0

        elif self.N_wall and self.W_wall:
            self.S += self.a_I_jplus1*v_N + self.a_Iminus1_j*v_W
            self.a_I_jplus1 = 0
            self.a_Iminus1_j = 0

        elif self.S_wall and self.E_wall:
            self.S += self.a_I_jminus1*v_S + MU*grad_vx_E*hy
            self.a_I_jminus1 = 0

        elif self.N_wall and self.E_wall:
            self.S += self.a_I_jplus1*v_N + MU*grad_vx_E*hy
            self.a_I_jplus1 = 0

#=========================== pressure nodes ======================================

class pressure_node():
    def __init__(self, node_num, u_prev, v_prev, d_u, d_v):

        self.N_wall = False
        self.S_wall = False
        self.E_wall = False
        self.W_wall = False

        #Check whether the node is at boundary or at corner
        self.is_boundary(node_num)
        self.is_corner(node_num)

        if self.W_wall:
            self.a_Iminus1_J = 0
        else:
            self.a_Iminus1_J = hy*d_u[node_num - 1 - idx_shft]

        self.a_Iplus1_J = hy*d_u[node_num - idx_shft]

        if self.N_wall:
            self.a_I_Jplus1 = 0
        else:
            self.a_I_Jplus1 = hx*d_v[node_num + int((node_num - EPS) // (Nx-1)) - idx_shft]

        if self.S_wall:
            self.a_I_Jminus1 = 0
        else:
            self.a_I_Jminus1 =  hx*d_v[node_num + int((node_num-EPS) // (Nx-1)) - Nx - idx_shft]

        self.a_I_J = self.a_Iminus1_J + self.a_Iplus1_J + self.a_I_Jplus1 + self.a_I_Jminus1

        # print(node_num, node_num - Nx + int((node_num-EPS) // (Nx-1)))

        if self.is_corner(node_num):
            self.updateCoeff_corner(node_num, u_prev, v_prev)
        elif self.is_boundary(node_num):
            self.updateCoeff_boundary(node_num, u_prev, v_prev)
        else:
            self.S = -((u_prev[node_num - idx_shft] - u_prev[node_num - 1 - idx_shft])*hy +
                       (v_prev[node_num + int((node_num - EPS) // (Nx-1)) - idx_shft] - v_prev[node_num + int((node_num-EPS) // (Nx-1)) - Nx - idx_shft])*hx)

    def is_boundary(self, node_num):

        if node_num > 1 and node_num < Nx - 1:
            self.S_wall = True
            return True
        elif node_num in [j*(Nx-1) + 1 for j in range(1, Ny-1)]:
            self.W_wall = True
            return True
        elif node_num in [(j+1)*(Nx-1) for j in range(1, Ny-1)]:
            self.E_wall = True
            return True
        elif node_num > (Nx-1)*Ny - (Nx-2) and node_num < (Nx-1)*Ny:
            self.N_wall = True
            return True

    def is_corner(self, node_num):

        if node_num == 1:
            self.S_wall = True
            self.W_wall = True
            return True
        elif node_num == Nx - 1:
            self.S_wall = True
            self.E_wall = True
            return True
        elif node_num == (Nx-1)*Ny - (Nx-2):
            self.N_wall = True
            self.W_wall = True
            return True
        elif node_num == (Nx-1)*Ny:
            self.N_wall = True
            self.E_wall = True
            return True

    def updateCoeff_boundary(self, node_num, u_prev, v_prev):

        if self.W_wall:
            self.S = -((u_prev[node_num - idx_shft] - U_0)*hy +
                       (v_prev[node_num + int((node_num-EPS) // (Nx-1)) - idx_shft] - v_prev[node_num + int((node_num-EPS) // (Nx-1)) - Nx - idx_shft])*hx)

        elif self.S_wall:
            self.S = -((u_prev[node_num - idx_shft] - u_prev[node_num - 1 - idx_shft])*hy +
                       (v_prev[node_num + int((node_num-EPS) // (Nx-1)) - idx_shft] - v_S)*hx)

        elif self.N_wall:
            self.S = -((u_prev[node_num - idx_shft] - u_prev[node_num - 1 - idx_shft])*hy +
                       (v_N - v_prev[node_num + int((node_num-EPS) // (Nx-1)) - Nx - idx_shft])*hx)

        elif self.E_wall:
            self.S = -((u_prev[node_num - idx_shft] - u_prev[node_num - 1 - idx_shft])*hy +
                       (v_prev[node_num + int((node_num-EPS) // (Nx-1)) - idx_shft] - v_prev[node_num + int((node_num-EPS) // (Nx-1)) - Nx - idx_shft])*hx)
            self.a_Iplus1_J = 0

    def updateCoeff_corner(self, node_num, u_prev, v_prev):

        if self.S_wall and self.W_wall:
            self.S = -((u_prev[node_num - idx_shft] - U_0)*hy +
                       (v_prev[node_num + int((node_num-EPS) // (Nx-1)) - idx_shft] - v_S)*hx)

        elif self.N_wall and self.W_wall:
            self.S = -((u_prev[node_num - idx_shft] - U_0)*hy +
                       (v_N - v_prev[node_num + int((node_num-EPS)// (Nx-1)) - Nx - idx_shft])*hx)

        elif self.S_wall and self.E_wall:
            self.S = -((u_prev[node_num - idx_shft] - u_prev[node_num - 1 - idx_shft])*hy +
                       (v_prev[node_num + int((node_num-EPS) // (Nx-1)) - idx_shft] - v_S)*hx)
            self.a_Iplus1_J = 0

        elif self.N_wall and self.E_wall:
            self.S = -((u_prev[node_num - idx_shft] - u_prev[node_num - 1 - idx_shft])*hy +
                       (v_N - v_prev[node_num + int((node_num-EPS) // (Nx-1)) - Nx - idx_shft])*hx)
            self.a_Iplus1_J = 0

'''
===================================================================================
                                FUNCTIONS
===================================================================================
'''

def L2norm(pred, actual):
    return np.sqrt(np.mean(np.square(pred - actual)))

def form_Mesh(Nx, Ny):

    x = np.linspace(0, L, Nx)
    y = np.linspace(0, H, Ny+1)

    ix = x/(L - 0)
    iy = y/(H - 0)

    #generate cartesian (or rectangular) grid
    north_boundary = iy[-1]*np.ones_like(ix)

    south_boundary = np.zeros_like(north_boundary)
    dY =  north_boundary - south_boundary

    #use the outer product to complete the 2D x-/y-coord
    x = np.outer(ix, np.ones_like(iy))
    y = np.outer(dY, iy) + np.outer(south_boundary, np.ones_like(iy))

    #scale for the graphics
    X=x*(L-0) + 0;  Y=y*(H-0) + 0

    return X, Y

def correct_u_vel(u_new, d_u, p_correction):

    u_corrected = np.zeros_like(u_new)

    for n in range(1, (Nx-1)*Ny + 1):
        if n in [(j+1)*(Nx-1) for j in range(0, Ny)]: #East Wall (including corners)
            u_corrected[n-1] = u_new[n-1] + d_u[n-1]*(p_correction[n-1])
        else:
            u_corrected[n-1] = u_new[n-1] + d_u[n-1]*(p_correction[n-1] - p_correction[n])

    return u_corrected

def correct_v_vel(v_new, d_v, p_correction):

    v_corrected = np.zeros_like(v_new)

    for n in range(1, Nx*(Ny-1) + 1):
        if n in [j*Nx + Nx for j in range(0, Ny-1)]: #East wall (including corners)
            v_corrected[n-1] = v_new[n-1]
        else:
            v_corrected[n-1] = v_new[n-1] + d_v[n-1]*(p_correction[n - int((n-EPS) // Nx) - idx_shft] - p_correction[n - int((n-EPS) // Nx) + (Nx-1) - idx_shft])

    return v_corrected

def SIMPLE(u_prev, v_prev, p_prev, p_correction_prev):

    '''
            Solve for u-velocity
    '''
    #Matrix formation for u-momemtum equation
    P_u = np.zeros(((Nx-1)*Ny, (Nx-1)*Ny))
    Q_u = np.zeros(((Nx-1)*Ny, ))

    for i in range(1, (Nx-1)*Ny + 1):

        node_u = u_vel_node(i, u_prev, v_prev, p_prev)

        P_u[i-1, i-1] = node_u.a_i_J/RELAXATION_FACTOR
        Q_u[i-1] = node_u.S + (1-RELAXATION_FACTOR)*node_u.a_i_J*u_prev[i-1]/RELAXATION_FACTOR

        if node_u.is_corner(i):

            if node_u.W_wall and node_u.S_wall:
                P_u[i-1, i] = -node_u.a_iplus1_J
                P_u[i-1, i-1 + (Nx-1)] = -node_u.a_i_Jplus1
            elif node_u.W_wall and node_u.N_wall:
                P_u[i-1, i] = -node_u.a_iplus1_J
                P_u[i-1, i-1 - (Nx-1)] = -node_u.a_i_Jminus1
            elif node_u.E_wall and node_u.S_wall:
                P_u[i-1, i-2] = -node_u.a_iminus1_J
                P_u[i-1, i-1 + (Nx-1)] = -node_u.a_i_Jplus1
            elif node_u.E_wall and node_u.N_wall:
                P_u[i-1, i-2] = -node_u.a_iminus1_J
                P_u[i-1, i-1 - (Nx-1)] = -node_u.a_i_Jminus1

            continue

        elif node_u.is_boundary(i):

            if node_u.W_wall:
                P_u[i-1, i] = -node_u.a_iplus1_J
                P_u[i-1, i-1 + (Nx-1)] = -node_u.a_i_Jplus1
                P_u[i-1, i-1 - (Nx-1)] = -node_u.a_i_Jminus1
            elif node_u.S_wall:
                P_u[i-1, i] = -node_u.a_iplus1_J
                P_u[i-1, i-2] = -node_u.a_iminus1_J
                P_u[i-1, i-1 + (Nx-1)] = -node_u.a_i_Jplus1
            elif node_u.N_wall:
                P_u[i-1, i] = -node_u.a_iplus1_J
                P_u[i-1, i-2] = -node_u.a_iminus1_J
                P_u[i-1, i-1 - (Nx-1)] = -node_u.a_i_Jminus1
            elif node_u.E_wall:
                P_u[i-1, i-2] = -node_u.a_iminus1_J
                P_u[i-1, i-1 + (Nx-1)] = -node_u.a_i_Jplus1
                P_u[i-1, i-1 - (Nx-1)] = -node_u.a_i_Jminus1

            continue

        else:
            P_u[i-1, i] = -node_u.a_iplus1_J
            P_u[i-1, i-2] = -node_u.a_iminus1_J
            P_u[i-1, i-1 + (Nx-1)] = -node_u.a_i_Jplus1
            P_u[i-1, i-1 - (Nx-1)] = -node_u.a_i_Jminus1

    u_new = np.linalg.solve(P_u, Q_u)
    d_u = np.divide(hy, np.diag(P_u))

    #Calculate u-momentum residual
    u_mom_res = 0
    for i in range((Nx-1)*Ny):
        u_mom_res += abs(np.dot(P_u[i], u_prev) - Q_u[i])

    '''
            Solve for v-velocity
    '''
    #Matrix formation for v-momentum equation
    P_v = np.zeros((Nx*(Ny-1), Nx*(Ny-1)))
    Q_v = np.zeros((Nx*(Ny-1), ))

    for i in range(1, Nx*(Ny-1) + 1):

        node_v = v_vel_node(i, u_prev, v_prev, p_prev)

        P_v[i-1, i-1] = node_v.a_I_j/RELAXATION_FACTOR
        Q_v[i-1] = node_v.S + (1-RELAXATION_FACTOR)*node_v.a_I_j*v_prev[i-1]/RELAXATION_FACTOR

        if node_v.is_corner(i):

            if node_v.W_wall and node_v.S_wall:
                P_v[i-1, i] = -node_v.a_Iplus1_j
                P_v[i-1, i-1 + Nx] = -node_v.a_I_jplus1
            elif node_v.W_wall and node_v.N_wall:
                P_v[i-1, i] = -node_v.a_Iplus1_j
                P_v[i-1, i-1 - Nx] = -node_v.a_I_jminus1
            elif node_v.E_wall and node_v.S_wall:
                P_v[i-1, i-2] = -node_v.a_Iminus1_j
                P_v[i-1, i-1 + Nx] = -node_v.a_I_jplus1
            elif node_v.E_wall and node_v.N_wall:
                P_v[i-1, i-2] = -node_v.a_Iminus1_j
                P_v[i-1, i-1 - Nx] = -node_v.a_I_jminus1

            continue

        elif node_v.is_boundary(i):

            if node_v.W_wall:
                P_v[i-1, i] = -node_v.a_Iplus1_j
                P_v[i-1, i-1 + Nx] = -node_v.a_I_jplus1
                P_v[i-1, i-1 - Nx] = -node_v.a_I_jminus1
            elif node_v.S_wall:
                P_v[i-1, i] = -node_v.a_Iplus1_j
                P_v[i-1, i-2] = -node_v.a_Iminus1_j
                P_v[i-1, i-1 + Nx] = -node_v.a_I_jplus1
            elif node_v.N_wall:
                P_v[i-1, i] = -node_v.a_Iplus1_j
                P_v[i-1, i-2] = -node_v.a_Iminus1_j
                P_v[i-1, i-1 - Nx] = -node_v.a_I_jminus1
            elif node_v.E_wall:
                P_v[i-1, i-2] = -node_v.a_Iminus1_j
                P_v[i-1, i-1 + Nx] = -node_v.a_I_jplus1
                P_v[i-1, i-1 - Nx] = -node_v.a_I_jminus1

            continue

        else:
            P_v[i-1, i] = -node_v.a_Iplus1_j
            P_v[i-1, i-2] = -node_v.a_Iminus1_j
            P_v[i-1, i-1 + Nx] = -node_v.a_I_jplus1
            P_v[i-1, i-1 - Nx] = -node_v.a_I_jminus1

    v_new = np.linalg.solve(P_v, Q_v)
    d_v = np.divide(hx, np.diag(P_v))

    #Calculate v-momentum residual
    v_mom_res = 0
    for i in range(Nx*(Ny-1)):
        v_mom_res += abs(np.dot(P_v[i], v_prev) - Q_v[i])

    '''
            Solve for continuity equation
    '''
    #Matrix formation for pressure-correction (continuity equation)
    P_p = np.zeros(((Nx-1)*Ny, (Nx-1)*Ny))
    Q_p = np.zeros(((Nx-1)*Ny, ))

    for i in range(1, (Nx-1)*Ny + 1):

        node_p = pressure_node(i, u_new, v_new, d_u, d_v)

        P_p[i-1, i-1] = node_p.a_I_J
        Q_p[i-1] = node_p.S

        if node_p.is_corner(i):

            if node_p.W_wall and node_p.S_wall:
                P_p[i-1, i] = -node_p.a_Iplus1_J
                P_p[i-1, i-1 + (Nx-1)] = -node_p.a_I_Jplus1
            elif node_p.W_wall and node_p.N_wall:
                P_p[i-1, i] = -node_p.a_Iplus1_J
                P_p[i-1, i-1 - (Nx-1)] = -node_p.a_I_Jminus1
            elif node_p.E_wall and node_p.S_wall:
                P_p[i-1, i-2] = -node_p.a_Iminus1_J
                P_p[i-1, i-1 + (Nx-1)] = -node_p.a_I_Jplus1
            elif node_p.E_wall and node_p.N_wall:
                P_p[i-1, i-2] = -node_p.a_Iminus1_J
                P_p[i-1, i-1 - (Nx-1)] = -node_p.a_I_Jminus1

            continue

        elif node_p.is_boundary(i):

            if node_p.W_wall:
                P_p[i-1, i] = -node_p.a_Iplus1_J
                P_p[i-1, i-1 + (Nx-1)] = -node_p.a_I_Jplus1
                P_p[i-1, i-1 - (Nx-1)] = -node_p.a_I_Jminus1
            elif node_p.S_wall:
                P_p[i-1, i] = -node_p.a_Iplus1_J
                P_p[i-1, i-2] = -node_p.a_Iminus1_J
                P_p[i-1, i-1 + (Nx-1)] = -node_p.a_I_Jplus1
            elif node_p.N_wall:
                P_p[i-1, i] = -node_p.a_Iplus1_J
                P_p[i-1, i-2] = -node_p.a_Iminus1_J
                P_p[i-1, i-1 - (Nx-1)] = -node_p.a_I_Jminus1
            elif node_p.E_wall:
                P_p[i-1, i-2] = -node_p.a_Iminus1_J
                P_p[i-1, i-1 + (Nx-1)] = -node_p.a_I_Jplus1
                P_p[i-1, i-1 - (Nx-1)] = -node_p.a_I_Jminus1

            continue

        else:
            P_p[i-1, i] = -node_p.a_Iplus1_J
            P_p[i-1, i-2] = -node_p.a_Iminus1_J
            P_p[i-1, i-1 + (Nx-1)] = -node_p.a_I_Jplus1
            P_p[i-1, i-1 - (Nx-1)] = -node_p.a_I_Jminus1

    p_correction = np.linalg.solve(P_p, Q_p)

    #Calculate Continuity residual
    con_res = 0
    for i in range((Nx-1)*Ny):
        con_res += abs(np.dot(P_p[i], p_correction_prev) - Q_p[i])

    #Check if any of the above matrix is diagonally dominant or NOT!
    # if not diagonally_dominant(P_u):
    #     raise Exception('P_u is NOT diagonally dominant')

    # if not diagonally_dominant(P_v):
    #     raise Exception('P_v is NOT diagonally dominant')

    # if not diagonally_dominant(P_p):
    #     raise Exception('P_p is NOT diagonally dominant')

    #Correct both u and v velocities
    u_corrected = correct_u_vel(u_new, d_u, p_correction)
    v_corrected = correct_v_vel(v_new, d_v, p_correction)

    return u_corrected, v_corrected, p_correction, u_mom_res, v_mom_res, con_res

def diagonally_dominant(X):
    D = np.diag(np.abs(X)) # Find diagonal coefficients
    S = np.sum(np.abs(X), axis=1) - D # Find row sum without diagonal

    if np.all(D - S >= 0):
        return True
    else:
        r = len(D)
        for row in range(r):
            if S[row] > D[row]:                
                print('Difference b/w sum of abs non-diagonal element values and abs diagoanl values for node num ' + str(row) + ' is:', D[row]-S[row])
        # print(X)
        return False

def exactSol(y_arr):
    
    #To compute fully developed laminar flow velocity profile between two fixed parallel plates, first we need to calculate pressure gradient
    neg_dpdx = H*U_0*12*H*MU    #(from volume flow rate relation)

    return (neg_dpdx/(2*MU*(H**2)))*(y_arr/H - np.square(y_arr)/(H**2))

'''
====================================================================================
                                USER INPUTS
====================================================================================
'''

# L = float(input("Enter the value of length in meters (L): "))
# H = float(input("Enter the value of height in meters (H): "))

Nx = int(input("Enter the number of u-velocity nodes to be considered (Nx): "))
Ny = int(input("Enter the number of v-velocity nodes to be considered (Ny): "))

U_0 = 0.02#0.34 #float(input("Enter the value of uniform velocity (U_0) in m/s: "))

RELAXATION_FACTOR = 0.6 #float(input("Enter the relaxation factor between 0 and 1: "))

'''
=====================================================================================
                         DOMAIN AND BOUNDARY CONDITIONS
=====================================================================================
'''

RHO = 910                  #density of water
MU = 1e-3#0.03094#1e-3                   #viscosity of water (approximately)

EPS = 1e-8
idx_shft = 1

H = 0.1

Re = RHO*U_0*H/MU           #Reynolds Number of the flow

L = 20*H*5

#Check for LAMINAR FLOW!!
not_laminar_err = "Enter input values such that Reynolds number is less than 2000 for flow to be laminar.\n"
if Re <= 2000:
    print("\nReynolds number is " + str(Re) + "; hence flow is laminar!\n")
else:
    print("\nReynolds number is " + str(Re))
    raise Exception(not_laminar_err)

Np = Nx                     #Number of pressure nodes
hx = 2*L/(2*Nx - 1)         #grid spacing in x-direction
hy = H/Ny                   #grid spacing in y-direction

v_W = 0                     #v velocity at West boundary
v_S = 0                     #v velocity at South boundary
v_N = 0                     #v velocity at North boundary
grad_vx_E = 0               #∂v/∂x at East boundary

u_S = 0                     #u velocity at South boundary
u_N = 0                     #u velocity at North boundary
grad_ux_E = 0               #∂u/∂x at East boundary

p_E = 0                     #static pressure at East boundary

'''
=======================================================================================
                INITIAL GUESS FOR PRESSURE AND VELOCITY FIELD
=======================================================================================
'''

p_initial = np.reshape([np.linspace(2, 0, Nx-1) for _ in range(Ny)], ((Nx-1)*Ny,))
u_initial = np.reshape([U_0*np.ones(Nx-1, ) for _ in range(Ny)], ((Nx-1)*Ny,))
v_initial = np.reshape([np.zeros(Nx, ) for _ in range(Ny-1)], ((Nx)*(Ny-1),))

'''
=======================================================================================
                            ITERATIVE SOLUTION
=======================================================================================
'''

#set tolerance value used for convergence criteria
TOLERANCE = 1e-5

con_res_vec = []

#store momentum residual per iteration
momentum_res = 100
mom_res_vec = []

u_prev = u_initial
v_prev = v_initial
p_prev = p_initial
p_corr_prev = np.zeros_like(p_prev)

N_iter = 0

while momentum_res > TOLERANCE:

    u_cal, v_cal, p_correction, u_mom_res, v_mom_res, continuity_residual = SIMPLE(u_prev, v_prev, p_prev, p_corr_prev)

    con_res_vec.append(continuity_residual)
    p_cal = p_prev + RELAXATION_FACTOR*p_correction

    momentum_res = max(u_mom_res, v_mom_res)

    mom_res_vec.append(momentum_res)

    # #use relaxation scheme
    # u_new = np.multiply((1 - RELAXATION_FACTOR), u_prev) + np.multiply(RELAXATION_FACTOR, u_cal)
    # v_new = np.multiply((1 - RELAXATION_FACTOR), v_prev) + np.multiply(RELAXATION_FACTOR, v_cal)
    # p_new = np.multiply((1 - RELAXATION_FACTOR), p_prev) + np.multiply(RELAXATION_FACTOR, p_cal)

    # print(max(p_cal), max(u_cal), max(v_cal))

    #use corrected fields as guess values for next iteration
    u_prev = u_cal
    v_prev = v_cal
    p_prev = p_cal
    p_corr_prev = p_correction

    #update the current number of iterations
    N_iter += 1

    print("Iteration number: " + str(N_iter) + " || Maximum momentum residual: " + str(momentum_res) + ' || Continuity Residual: ' + str(continuity_residual))

'''
=======================================================================================
                            CONTOURS AND PLOTS
=======================================================================================
'''

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(16, 12))
fig.suptitle('For Nx = ' + str(Nx) + ', Ny = ' + str(Ny) + ', L = ' + str(L) + ', H = ' + str(H) + r', $U_{0}$ = ' + str(U_0))
fig.subplots_adjust(hspace=0.5)

#Plotting u-velocity contour
west_wall_indices = [j*(Nx-1) for j in range(0, Ny)]
u_complete = np.insert(u_prev, west_wall_indices, U_0)

x_c = [j*hx for j in range(0, Nx)]
y_c = [j*hy + hy/2.0 for j in range(Ny)]
xv, yv = np.meshgrid(x_c, y_c)

ax1.contourf(xv, yv, np.reshape(u_complete, (Ny, Nx)), levels=1000, cmap='rainbow', vmin=np.min(u_complete), vmax=np.max(u_complete))
ax1.set_xlabel("L")
ax1.set_ylabel("H")
ax1.set_title(r"$u(x,y)$")
plt.colorbar(ax1.contourf(xv, yv, np.reshape(u_complete, (Ny, Nx)), levels=1000, cmap='rainbow', vmin=np.min(u_complete), vmax=np.max(u_complete)), ax=ax1)

#Plotting v-velocity contour
v_prev = np.insert(v_prev, 0, np.zeros((Nx,)))
v_complete = np.concatenate((v_prev, np.zeros((Nx,))), axis=0)

x_c = [j*hx + hx/2.0 for j in range(0, Nx)]
y_c = [j*hy for j in range(Ny+1)]
xv, yv = np.meshgrid(x_c, y_c)

ax2.contourf(xv, yv, np.reshape(v_complete, (Ny+1, Nx)), levels=1000, cmap='rainbow', vmin=np.min(v_complete), vmax=np.max(v_complete))
ax2.set_xlabel("L")
ax2.set_ylabel("H")
ax2.set_title(r"$v(x,y)$")
plt.colorbar(ax2.contourf(xv, yv, np.reshape(v_complete, (Ny+1, Nx)), levels=Nx, cmap='rainbow', vmin=np.min(v_complete), vmax=np.max(v_complete)), ax=ax2)

#Plotting pressure contour
east_wall_indices = [j*(Nx-1) + Nx-2 for j in range(0, Ny)]
p_complete = np.insert(p_prev, east_wall_indices, p_E)

x_c = [j*hx + hx/2.0 for j in range(0,Nx)]
y_c = [j*hy + hy/2.0 for j in range(Ny)]
xv, yv = np.meshgrid(x_c, y_c)

ax3.contourf(xv, yv, np.reshape(p_complete, (Ny, Nx)), levels=Nx, cmap='rainbow', vmin=np.min(p_complete), vmax=np.max(p_complete))
ax3.set_xlabel("L")
ax3.set_ylabel("H")
ax3.set_title(r"$p(x,y)$")
plt.colorbar(ax3.contourf(xv, yv, np.reshape(p_complete, (Ny, Nx)), levels=Nx, cmap='rainbow', vmin=np.min(p_complete), vmax=np.max(p_complete)), ax=ax3)

# X, Y = form_Mesh(Nx, Ny)
# f = np.reshape(u_new, (Ny, Nx-1))
# plt.figure()
# plt.pcolormesh(X.T, Y.T, f, edgecolors='k', lw=0.5, cmap='rainbow', vmin=np.min(u_new), vmax=np.max(u_new))
# plt.xlim([0, L])
# plt.ylim([0, H])
# plt.xlabel("L")
# plt.ylabel("H")
# plt.title(r"$u(x,y)$")
# plt.colorbar()

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
fig.suptitle('For Nx = ' + str(Nx) + ', Ny = ' + str(Ny) + ', L = ' + str(L) + ', H = ' + str(H) + r', $U_{0}$ = ' + str(U_0))
fig.subplots_adjust(wspace=0.5)

ax1.plot(con_res_vec)
ax1.set_xlabel('Number of iterations')
ax1.set_ylabel('Continuity Residual')

x_c = np.array([j*hx for j in range(0, Nx)])
y_c = [j*hy + hy/2.0 for j in range(Ny)]
y_c = np.concatenate(([0], y_c, [H]))

Lby4_idx = (np.abs(x_c - L/4.0)).argmin()
Lby2_idx = (np.abs(x_c - L/2.0)).argmin()
Lby1_idx = (np.abs(x_c - L/1.0)).argmin()

u_mat = u_complete.reshape((Ny, Nx))
u_Lby4 = np.concatenate(([0], u_mat[:, Lby4_idx], [0]))
u_Lby2 = np.concatenate(([0], u_mat[:, Lby2_idx], [0]))
u_Lby1 = np.concatenate(([0], u_mat[:, Lby1_idx], [0]))

ax2.plot(u_Lby4, y_c, label='x=L/4')
ax2.plot(u_Lby2, y_c, label='x=L/2')
ax2.plot(u_Lby1, y_c, label='x=L')
ax2.plot(exactSol(y_c), y_c, label='Exact Solution')
ax2.set_xlabel('u(y)')
ax2.set_ylabel('y')
ax2.legend()


plt.show()