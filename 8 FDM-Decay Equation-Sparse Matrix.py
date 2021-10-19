import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse import linalg


def main():

    print("Solving the nuclear decay equation")

    t12 = float(input("Half life = "))

    c = math.log(2) / t12

    st = float(input("Start of time interval = "))

    et = float(input("End of time interval = "))

    h = float(input("Step size = "))            # distance between grid points

    t = np.arange(start=st, stop=et, step=h, dtype=float)  # time array

    n = t.size

    A = sp.lil_matrix((n, n))               # Coefficient matrix

    A.setdiag(-1/(2 * h), -1)                   # Building the diagonals
    A.setdiag(c, 0)
    A.setdiag(1/(2 * h), 1)

    A[0, 0] = 1                                 # Building the 0th row
    A[0, 1] = 0

    A[n - 1, n - 2] = -1 / h                        # Building the nth row
    A[n - 1, n - 1] = 1 / h + c

    A = A.tocsr()

    b = sp.lil_matrix((n, 1))               # RHS vector

    b[0, 0] = float(input("Initial mass = "))

    y = sp.linalg.spsolve(A, b)                 # Solving the system

    plt.plot(t, y, label="dm/dt = - (ln|2| / half-life) * m")          # Plotting the system
    plt.title("Decay Equation")
    plt.xlabel("time")
    plt.ylabel("mass")
    plt.legend()
    plt.show()


main()
