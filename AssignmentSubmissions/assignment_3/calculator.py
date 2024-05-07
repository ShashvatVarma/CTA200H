import numpy as np

def computation(grid_density):
    #create a two dimensional grid between
    #-2 and 2, with stepsize of 0.2
    x = np.linspace(-2, 2, grid_density, endpoint=True)
    y = np.linspace(-2, 2, grid_density, endpoint=True)

    #define the c=x+iy plane given the defined arrays
    C = np.zeros((len(x), len(y)), dtype=complex)
    for i in range(len(x)):
        for j in range(len(y)):
            C[i,j] = x[i] + 1j*y[j]

    #define the function f(z) = z^2 + c
    def f(z, c):
        return abs(z)**2 + c


    #to plot the grid, we may use the real() and imag() functions
    iteration_break = []
    broken_x = [] #these arrays contain locations of the points that diverge
    broken_y = []
    bound_x = []
    bound_y = []
    for i in range(len(x)):
        for j in range(len(y)):
            value = C[i,j]

            #now we will check whether the seqeuence Zn+1 = Zn^2 + c diverges given this value of c, where the square is
            #modulus
            z=0
            exited=False
            for p in range(100):
                new_z = f(z, value)
                z = new_z
                if abs(new_z) > 30:
                    iteration_break.append(p)
                    broken_x.append(np.real(value))
                    broken_y.append(np.imag(value))
                    exited=True
                    break
            if not exited:
                bound_x.append(np.real(value))
                bound_y.append(np.imag(value))        
    return iteration_break, broken_x, broken_y, bound_x, bound_y