from sympy.ntheory import discrete_log
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def residues_to_powers(residues, mod, generator):
    powers = []
    for r in residues:
        power = discrete_log(mod, r, generator)
        powers.append(power)
    return powers

def log_residues(residues, mod, generator):
	# residues += [mod - r for r in residues]
	# residues = list(set(residues))
	powers = residues_to_powers(residues, mod, generator)
	return powers

	
NinasLists = [[6,16,18], [10,14,16], [5,8,12,16], [1,5,9,17], [3,9,13,14], [2,3,8,17], [1,7,11,16], [7,14,17,18], [1,8,18], [4,12,18]]
NinasLists = [[r % 18 for r in log_residues(residues, 37, 2)] for residues in NinasLists]

def plot_single_circle(ax, residues, mod):
	mod = (mod-1)//2
	ax.set_aspect('equal', 'box')
	ax.set_xlim(-1.5, 1.5)
	ax.set_ylim(-1.5, 1.5)

	# Draw the circle
	circle = Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
	ax.add_artist(circle)

	# Draw the residues as dots
	for residue in residues:
		angle = 2 * np.pi * residue / (mod)
		x = np.cos(angle)
		y = np.sin(angle)
		ax.plot(x, y, 'o', color='red')
	missing_residues = set(range(mod)) - set(residues)
	for residue in missing_residues:
		angle = 2 * np.pi * residue / mod
		x = np.cos(angle)
		y = np.sin(angle)
		ax.plot(x, y, 'o', color='gray')

    
# residues = [1, 5, 9, 13]  # Replace this with your list of residues mod 36
# mod = 37
# plot_residues_on_circle(residues, mod)


def plot_residues_on_grid(residue_lists, grid_shape=(2, 5)):
    fig, axs = plt.subplots(*grid_shape)
    mod = 37
    
    for ax, residues in zip(axs.ravel(), residue_lists):
        plot_single_circle(ax, residues, mod)
    
    plt.show()

# # Example residue lists
# res1 = [1, 4, 7]
# res2 = [2, 8, 14]
# res3 = []
# res4 = []
# res5 = []
# res6 = []
# res7 = []
# res8 = []
# res9 = []
# res10 = [3, 6, 9]

# # Create a list containing all the residue lists
# all_residues = [res1, res2, res3, res4, res5, res6, res7, res8, res9, res10]
print(NinasLists)
plot_residues_on_grid(NinasLists)











