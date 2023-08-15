import numpy as np
import matplotlib.pyplot as plt

# Create a meshgrid of points in the specified range
x = np.linspace(-3, 3, 1000)
y = np.linspace(-3, 3, 1000)
x, y = np.meshgrid(x, y)

# Define the function for the level sets
z = x**2 * y**2 + 0.1 * (1 - np.exp(0-x**2 - y**2))

# Plot the contours for the specified levels
levels = [0, 0.05, 0.1, 0.2, 99]
colors = ['red', 'green', 'blue', 'black', 'purple']
plt.contourf(x, y, z, levels=levels, colors=colors, alpha=0.5)

# Add a colorbar to indicate the levels
plt.colorbar(ticks=levels, label='Level')

plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Level sets of $f(x, y) = x^2 \cdot y^2 + 0.1(1 - \exp(-x^2 - y^2))$')
plt.savefig('level_fruit_plot.png')
plt.show()
