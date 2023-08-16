import numpy as np
import matplotlib.pyplot as plt

# Create a meshgrid of points in the specified range
x = np.linspace(-3, 3, 1000)
y = np.linspace(-3, 3, 1000)
x, y = np.meshgrid(x, y)

# Define the function for the level set
z = x**2 * y**2

# Plot the contour where the function is less than 0.01
levels = [0, 0.01, 0.05, 0.1, 99]
colors = ['red', 'green', 'blue', 'black', 'purple']
plt.contourf(x, y, z, levels=levels, colors=colors, alpha=0.5)
plt.colorbar(ticks=levels, label='Level')

# Optionally, you can plot the contour line where the function is exactly 0.01
#plt.contour(x, y, z, levels=[0.001], colors=['red'])

plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Level sets of $x^2 \cdot y^2$')
plt.savefig('slt_level.png')
plt.show()
