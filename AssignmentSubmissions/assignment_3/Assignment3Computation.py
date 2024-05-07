import numpy as np
import matplotlib.pyplot as plt
import calculator as calc

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
