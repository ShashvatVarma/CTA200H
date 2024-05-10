import numpy as np
import matplotlib.pyplot as plt
import calculator as calc
import scipy.integrate as spi

#plot the grid
#do the computation
iteration_break, broken_x, broken_y, bound_x, bound_y = calc.computation(100)
pointsize = 5
fig,ax = plt.subplots()
plt.scatter(bound_x, bound_y, color='green', s=pointsize, marker='s')
plt.scatter(broken_x, broken_y, c=iteration_break,cmap='inferno',vmin= np.min(iteration_break), vmax=np.max(iteration_break) , s=pointsize, marker='s')
cbar = plt.colorbar()
cbar.set_label("Iterations to Diverge")
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.show()

#define the Lorenz attractor ODE's
def xdot(x,y,z,sigma):
    return -sigma*(x-y)
def ydot(x,y,z,r):
    return -x*z + r*x - y
def zdot(x,y,z,b):
    return x*y - b*z

#solve the system of ODE's given some initial conidtios on the coefficients and the variables. 
sp.solve_ivp(fun = lambda t, y: [xdot(y[0],y[1],y[2],10), ydot(y[0],y[1],y[2],28), zdot(y[0],y[1],y[2],8/3)], t_span = (0,100), y0 = [1,1,1], method='RK45', t_eval = np.linspace(0,100,1000))