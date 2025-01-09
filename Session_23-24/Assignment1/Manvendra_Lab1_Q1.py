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

def idata(i, N, x_l, x_u, u_start, u_end):
    #Defining a function which returns x coordinate, and the constants coefficients which will be used in Ax=B for the ith grid point
    #Note that i will vary from 1,2,.....,N-2 for N grid points

    #Value of h (uniform grid spacing) 
    h = (x_u - x_l)/(N-1)

    #x-coordinate of ith grid point
    xi = x_l + i*h

    #Coefficient of u_{i-1} for ith grid point in coefficient matrix A
    ai = 2*xi - h               #or (xi/h**2) - (1/(2*h))

    #Coefficient of u_{i} for ith grid point in coefficient matrix A
    bi = 2*xi*(h**2) - 4*xi     #or xi - (2*xi/h**2)

    #Coefficient of u_{i+1} for ith grid point in coefficient matrix A
    ci = 2*xi + h               #or (xi/h**2) + (1/(2*h))

    #Coefficient of B_{i} for ith grid point in column matrix B
    if i == 1:
        Bi = -2*(h**2)*xi*(4*np.sin(xi) + 5*xi*np.cos(xi)) - ai*u_start
    elif i == N-2:
        Bi = -2*(h**2)*xi*(4*np.sin(xi) + 5*xi*np.cos(xi)) - ci*u_end
    else:
        Bi = -2*(h**2)*xi*(4*np.sin(xi) + 5*xi*np.cos(xi))

    return xi, ai, bi, ci, Bi

def genData(N, x_l, x_u, u_start, u_end):
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
        xi, ai, bi, ci, Bi = idata(i, N, x_l, x_u, u_start, u_end)
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

def matrixInv(N, x_l, x_u, u_start, u_end):
    #We can solve by making whole matrix A and finding its inverse

    #Generating the data
    x, mainDiag, subDiag, superDiag, B = genData(N, x_l, x_u, u_start, u_end)

    #Creating coefficient matrix A using diagonal vectors
    A = np.diag(mainDiag) + np.diag(superDiag, 1) + np.diag(subDiag, -1)

    #Finding inv(A)*B using inbuilt numpy library
    u = np.linalg.solve(A, B)

    #For complete solution, adding u at 0 and pi/2 at start and end
    u = np.insert(u, 0, u_start)
    u = np.append(u, u_end)

    return u, x

def triDiagAlgo(N, x_l, x_u, u_start, u_end):
    #For larger value of N, it will be sparse matrix. Tri-diagonal matrix algorithm is utilizes the sparity of matrix A to solve the system Ax=B
    
    #Generating the data
    x, mainDiag, subDiag, superDiag, B = genData(N, x_l, x_u, u_start, u_end)

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

def errRMS(N_vec, x_l, x_u, u_start, u_end):
    #In this function, we will calculate root mean squared error for different values of N (number of grid points)

    #Inititating error vector
    err = []

    for n in N_vec:
        #Obtain the numerical solution for different value of grid points and store the solution & x-vector
        u_N, x_N = triDiagAlgo(n, x_l, x_u, u_start, u_end)

        #For different values of N, length x-vector will be different and so of exact solution vector
        u_exact_N = [(-i**2)*np.sin(i) for i in x_N]

        #Obtaining root mean squre error for N grid points
        errRMS_N = mean_squared_error(u_exact_N, u_N, squared = False)
        err.append(errRMS_N)

    return err

#==========================DOMAIN AND BOUNDARY CONDITIONS============================
#Defining x-domain (start and end points)
x_l = 0
x_u = np.pi/2

#Boundaries Conditions at x=0 and x=pi/2
u_start = 0
u_end = -(np.pi**2)/4

#Number of grid points
N = 21

#===================================METHOD 1============================================
start_time1 = time.perf_counter()

u1, x = matrixInv(N, x_l, x_u, u_start, u_end)

print("Using matrix inversion\n--- %s seconds ---" % (time.perf_counter() - start_time1))

#===================================METHOD 2=============================================
start_time2 = time.perf_counter()

u2, x = triDiagAlgo(N, x_l, x_u, u_start, u_end)

print("Using tri-diagonal matrix alogrithm\n--- %s seconds ---" % (time.perf_counter() - start_time2))

#================================EXACT SOLUTION========================================

u_exact = [(-i**2)*np.sin(i) for i in x]

#=========================SOLUTION COMPARISON PLOTS======================================

#Initializing subplot
fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (15, 15))
plt.subplots_adjust(hspace=0.3)

#Creating subplots variable
ax1 = ax[0,0]
ax2 = ax[0,1]
ax3 = ax[1,0]
ax4 = ax[1,1]

#Plotting and comparing Numerical and Exact Solution 
ax1.set_xlabel('x')
ax1.set_ylabel("u(x)")
ax1.grid(which = 'major')
ax1.set_title("Numerical Solution of u at N = %d grid points"  %N)
#ax1.plot(x, u1, label='Numerical1')                             #Using matrix inversion
ax1.plot(x, u2, label='Numerical')                               #Tri-Diagonal Algorithm   
ax1.plot(x, u_exact, label='Exact')
ax1.legend()

#Plotting absolute error at every x position
ax2.set_xlabel("x")
ax2.set_ylabel(r"$\epsilon_{abs} = \left| u_{exact} - u_{numerical} \right|$")
ax2.grid(which = 'major')
ax2.set_title(r"$\epsilon_{abs}$ of u at N = %d grid points"  %N)
ax2.plot(x, abs(u2 - u_exact))

#Comments on above plot
print("From the graph it is verified that order of error is of h^2. Also it is observed that there are both negative and positive error")

#========================DIFFERENT VALUE OF NUMBER OF GRID POINTS========================

#Vector of different values of grid points
N_vec = [11, 21, 41, 81]

#Calculating vector of h for different values of N, where h is the grid spacing
GridSpace_vec = [(x_u - x_l)/(n-1) for n in N_vec]

#Calculating root mean square error for different solution corresponding to different values of N using user-defined function
err_RMS_vec = errRMS(N_vec, x_l, x_u, u_start, u_end)

#===================================SOME MORE PLOTS====================================

#Plotting root mean square vs h
ax3.set_xlabel(r"$h$")
ax3.set_ylabel(r"$\epsilon_{RMS}$")
ax3.grid(which = 'major')
ax3.set_title(r"$\epsilon_{RMS}$ as function of h")
ax3.plot(GridSpace_vec, err_RMS_vec)

#Obtaining logarithmic root mean square error using root mean square, similarly for grid spacing vector of h
logGridSpace_vec = abs(np.log10(GridSpace_vec))
err_logRMS_vec = abs(np.log10(err_RMS_vec))

#Plotting absolute logarithmic root mean square vs log(h) 
ax4.set_xlabel(r"$\left| \log(h) \right|$")
ax4.set_ylabel(r"$\left| \log(\epsilon_{RMS}) \right|$")
ax4.grid(which = 'major')
ax4.set_title(r"$\left| \log(\epsilon_{RMS}) \right|$ vs $\left| \log(h) \right|$")
ax4.plot(logGridSpace_vec, err_logRMS_vec)

#Slope of log plot 
orderOfAcc = (err_logRMS_vec[1] - err_logRMS_vec[0])/(logGridSpace_vec[1] - logGridSpace_vec[0])
print("Slope of log plot: ", orderOfAcc)

plt.show()