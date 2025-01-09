#==================================================================================#
#                                    APL 720                                       # 
#                      COMPUTATIONAL FLUID DYNAMICS LABORATORY                     # 
#                                  ASSIGNMENT 1                                    # 
#                      Submitted by: Manvendra Singh Rajawat                       # 
#                           Entry Number: 2023AMA2568                              # 
#==================================================================================#

#=====================================LIBRARIES=====================================

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error

#====================================FUNCTIONS=======================================

def idata(i, N, eps, x_l, x_u, u_start, u_end, method=1):
    #Defining a function which returns x coordinate, and the constants coefficients which will be used in Ax=B for the ith grid point
    #Note that i will vary from 1,2,.....,N-2 for N grid points

    #Value of h (uniform grid spacing) 
    h = (x_u - x_l)/(N-1)

    #x-coordinate of ith grid point
    xi = x_l + i*h

    #METHOD 1: Second order Central Difference approximation for both u" and u'
    if method == 1:
        #Coefficient of u_{i-1} for ith grid point in coefficient matrix A
        ai = 2*eps + h             

        #Coefficient of u_{i} for ith grid point in coefficient matrix A
        bi = -4*eps                

        #Coefficient of u_{i+1} for ith grid point in coefficient matrix A
        ci = 2*eps - h             

        #Coefficient of B_{i} for ith grid point in column matrix B
        if i == 1:
            Bi = -2*(h**2) - ai*u_start
        elif i == N-2:
            Bi = -2*(h**2) - ci*u_end
        else:
            Bi = -2*(h**2)

    #METHOD 2: Second order Central Difference approximation for u" and First order Backward Difference approximation for u'
    else: 
        #Coefficient of u_{i-1} for ith grid point in coefficient matrix A
        ai = eps + h             

        #Coefficient of u_{i} for ith grid point in coefficient matrix A
        bi = -(2*eps + h)                

        #Coefficient of u_{i+1} for ith grid point in coefficient matrix A
        ci = eps            

        #Coefficient of B_{i} for ith grid point in column matrix B
        if i == 1:
            Bi = -(h**2) - ai*u_start
        elif i == N-2:
            Bi = -(h**2) - ci*u_end
        else:
            Bi = -(h**2)

    return xi, ai, bi, ci, Bi

def genData(N, eps, x_l, x_u, u_start, u_end, method):
    #After discretizing the governing equation, we will get a set of algebraic equations, which can be written in form of Ax=B.
    #The Coefficient matrix wil be tri-diagonal matrix (only the entries in the main and the adjoining diagonals are non-zero)

    #Initialize 3 different vectors of diagonal, super-diagonal & sub-diagonal elements of coefficient matrix A
    mainDiag = []
    subDiag = []
    superDiag = []

    #Initialize column vector B
    B = []

    #Initialize x-vector
    x = [x_l]

    #Extracting values of all main/super/sub-diagonal as well as x and B elements using function defined above
    for i in range(1, N-1):
        xi, ai, bi, ci, Bi = idata(i, N, eps, x_l, x_u, u_start, u_end, method)
        mainDiag.append(bi)
        subDiag.append(ai)
        superDiag.append(ci)
        B.append(Bi)
        x.append(xi)

    #Adding x_u in the end to complete the x-vector
    x.append(x_u)

    #Since Boundary Values are known and are written in vector B, subDiag & superDiag will have one less element
    del subDiag[0]
    del superDiag[-1]

    return x, mainDiag, subDiag, superDiag, B

def matrixInv(N, eps, x_l, x_u, u_start, u_end, method):
    #We can solve by making whole matrix A and finding its inverse

    #Generating the data
    x, mainDiag, subDiag, superDiag, B = genData(N, eps, x_l, x_u, u_start, u_end, method)

    #Creating coefficient matrix A using diagonal vectors
    A = np.diag(mainDiag) + np.diag(superDiag, 1) + np.diag(subDiag, -1)

    #Finding inv(A)*B using inbuilt numpy library
    u = np.linalg.solve(A, B)

    #For complete solution, adding u at 0 and pi/2 at start and end
    u = np.insert(u, 0, u_start)
    u = np.append(u, u_end)

    return u, x

def triDiagAlgo(N, eps, x_l, x_u, u_start, u_end, method):
    #For larger value of N, it will be sparse matrix. Tri-diagonal matrix algorithm is utilizes the sparity of matrix A to solve the system Ax=B
    
    #Generating the data
    x, mainDiag, subDiag, superDiag, B = genData(N, eps, x_l, x_u, u_start, u_end, method)

    #Applying tri-diagonal matrix algorithm
    for i in range(1, N-2):
        mainDiag[i] = mainDiag[i] - (subDiag[i-1]/mainDiag[i-1])*superDiag[i-1]
        B[i] = B[i] - (subDiag[i-1]/mainDiag[i-1])*B[i-1]

    #Initializing solution vector u
    u = (B[-1]/mainDiag[-1])*np.ones((N-2,))

    #Back substitution
    for i in range(N-4,-1,-1):
        u[i] = (B[i] - superDiag[i]*u[i+1])/mainDiag[i]

    #For complete solution, adding u at 0 and pi/2 at start and end
    u = np.insert(u, 0, u_start)
    u = np.append(u, u_end)

    return u, x

def exactSol(X, eps):
    u = [1 + x + (np.exp(x/eps)-1)/(np.exp(1/eps)-1) for x in X]
    return u

def errRMS(N_vec, eps, x_l, x_u, u_start, u_end, method):
    #In this function, we will calculate root mean squared error for different values of N (number of grid points)

    #Inititating error vector
    err = []

    for n in N_vec:
        #Obtain the numerical solution for different value of grid points and store the solution & x-vector
        u_N, x_N = triDiagAlgo(n, eps, x_l, x_u, u_start, u_end, method)

        #For different values of N, length x-vector will be different and so of exact solution vector
        u_exact_N = exactSol(x_N, eps)

        #Obtaining root mean square error for N grid points
        errRMS_N = mean_squared_error(u_exact_N, u_N, squared = False)
        err.append(errRMS_N)

    return err

#==================================DOMAIN AND BOUNDARY CONDITIONS================================

#Defining x-domain (start and end points)
x_l = 0
x_u = 1

#Boundaries Conditions at x=0 and x=pi/2
u_start = 1
u_end = 3

#Number of grid points
N_vec = [11, 26, 101]

#Different Values of epsilon
EPS = [0.3, 0.1, 0.05, 0.005]

#===================================SOLUTION COMPARISON PLOTS============================================

for N in N_vec:

    #Initialising the subplot
    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (15, 15))
    plt.subplots_adjust(hspace = 0.3)

    for idx, eps in enumerate(EPS):
        #Calculate solution vector using both Method 1 and Method 2
        u1, x = triDiagAlgo(N, eps, x_l, x_u, u_start, u_end, method=1)
        u2, x = triDiagAlgo(N, eps, x_l, x_u, u_start, u_end, method=2)

        #Find exact solution
        u_exact = exactSol(x, eps)

        #Selecting the subplot
        axi = ax[int(idx/2), idx%2]

        #Plotting and comparing the numerical solution obtained using both methods, with exact solution
        axi.set_xlabel("x")
        axi.set_ylabel("u(x)")
        axi.grid(which = 'major')
        axi.set_title("Solution of u(x) with h = " + str((x_u-x_l)/(N-1)) + r" and $\epsilon$ = " + str(eps))
        axi.plot(x, u1, label='Numerical (Method 1)')                               
        axi.plot(x, u2, label='Numerical (Method 2)')                                  
        axi.plot(x, u_exact, label='Exact')
        axi.legend()

#=====================================ERROR PLOTS====================================================

#Calculating vector of h for different values of N, where h is the grid spacing
GridSpace_vec = [(x_u - x_l)/(n-1) for n in N_vec]

#Calculating absolute of logarithmic vector of grid space vector h
logGridSpace_vec = abs(np.log10(GridSpace_vec))

#Initializing subplot for plotting root mean square for different value of step size h
fig1, ax1 = plt.subplots(nrows = 2, ncols = 2, figsize = (15, 15))
plt.subplots_adjust(hspace = 0.3)

#Initializing subplot for plotting logarithmic root mean square for different value of step size h
fig2, ax2 = plt.subplots(nrows = 2, ncols = 2, figsize = (15, 15))
plt.subplots_adjust(hspace = 0.3)

for idx, eps in enumerate(EPS):

    #Calculating root mean square error for different solution corresponding to different values of N using user-defined function
    err_RMS_vec1 = errRMS(N_vec, eps, x_l, x_u, u_start, u_end, method=1)
    err_RMS_vec2 = errRMS(N_vec, eps, x_l, x_u, u_start, u_end, method=2)

    #Selecting the subplot
    ax1i = ax1[int(idx/2), idx%2]

    #Plotting root mean square error for different value of epsilon
    ax1i.set_xlabel(r"$h$")
    ax1i.set_ylabel(r"$e_{RMS}$")
    ax1i.grid(which = 'major')
    ax1i.set_title(r"$e_{RMS}$ vs h for $\epsilon$ = " + str(eps))
    ax1i.plot(GridSpace_vec, err_RMS_vec1, label = 'Method 1')
    ax1i.plot(GridSpace_vec, err_RMS_vec2, label = 'Method 2')
    ax1i.legend()

    #Calculating absolute logarithmic root mean square error using root mean square error
    err_logRMS_vec1 = abs(np.log10(err_RMS_vec1))
    err_logRMS_vec2 = abs(np.log10(err_RMS_vec2))

    #Selecting the subplot
    ax2i = ax2[int(idx/2), idx%2]

    #Plotting absolute logarithmic root mean square error for different value of epsilon
    ax2i.set_xlabel(r"$\left| \log(h) \right|$")
    ax2i.set_ylabel(r"$\left| \log(e_{RMS}) \right|$")
    ax2i.grid(which = 'major')
    ax2i.set_title(r"$\left| \log{e_{RMS}} \right|$ vs $\left| \log{h} \right|$ for $\epsilon = $" + str(eps))
    ax2i.plot(logGridSpace_vec, err_logRMS_vec1, label = 'Method 1')
    ax2i.plot(logGridSpace_vec, err_logRMS_vec2, label = 'Method 2')
    ax2i.legend()

    #Slope of log plot will provide order of accuracy
    orderOfAcc1 = (err_logRMS_vec1[1] - err_logRMS_vec1[0])/(logGridSpace_vec[1] - logGridSpace_vec[0])
    orderOfAcc2 = (err_logRMS_vec2[1] - err_logRMS_vec2[0])/(logGridSpace_vec[1] - logGridSpace_vec[0])
    print("Slope of log plot for epsilon = " + str(eps) + " using Method 1 is: ", orderOfAcc1)
    print("Slope of log plot for epsilon = " + str(eps) + " using Method 2 is: ", orderOfAcc2)

plt.show()
