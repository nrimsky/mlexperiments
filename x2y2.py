# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Define the function
# def func(x, y):
#     return x**2 * y**2

# # Create a meshgrid for the x and y values
# x = np.linspace(-2, 2, 100)
# y = np.linspace(-2, 2, 100)
# x, y = np.meshgrid(x, y)

# # Compute the corresponding z values
# z = func(x, y)

# # Create the 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x, y, z, cmap='viridis')

# # Label the axes
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # Show the plot
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import math

LEN = 2

def silu(x):
    return x / (1.0 + np.exp(-x))
# Define the function
def func(x, y):
    return -np.exp(0.5*(2-x**2 * y**2)) - 0.5*np.exp(0-x**2-y**2)
#    return -np.exp(0.3*(2-x**2*y**2))

# Create a meshgrid for the x and y values
x = np.linspace(-LEN, LEN, 400)
y = np.linspace(-LEN, LEN, 400)
x, y = np.meshgrid(x, y)

# Compute the corresponding z values
z = func(x, y)

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Custom normalization to emphasize lower z values
norm = mcolors.Normalize(vmin=z.min(), vmax=z.max(), clip=False)
surface = ax.plot_surface(x, y, z, cmap='viridis', norm=norm, rstride=10, cstride=10)

# Label the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Plot a cross at the minima (z=0)
# Plot a red cross at the minima (z=0), slightly above the surface
z_offset = -2  # Small offset to raise the cross above the surface
cross_length = LEN
cross_x = np.linspace(-cross_length, cross_length, 100)
cross_y = np.zeros_like(cross_x)
cross_z_x = func(cross_x, cross_y) + z_offset
cross_z_y = func(cross_y, cross_x) + z_offset
ax.plot(cross_x, cross_y, cross_z_x, color='#EE4B2B', lw=2)
ax.plot(cross_y, cross_x, cross_z_y, color='#EE4B2B', lw=2)

# Add a color bar
fig.colorbar(surface, ax=ax)

#save the plot
plt.savefig('x2y2actual.png')
# plt.savefig('x2y2.png')


# Show the plot
plt.show()
