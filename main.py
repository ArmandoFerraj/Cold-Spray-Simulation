import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import matplotlib.animation as animation

# Define the grid of x and y values
x = np.linspace(-10, 40, 200)  # Extended range
y = np.linspace(-10, 21, 200)
X, Y = np.meshgrid(x, y)
points = np.dstack((X, Y))

scale = 2
sigma = 2.25
covariance_matrix = [[sigma, 0], [0, sigma]]  # Covariance matrix

# Number of layers
num_layers = 3

# Create a figure and 3D axis for the plot
fig = plt.figure(facecolor='grey')
ax = fig.add_subplot(projection='3d', facecolor = 'grey')

# Pre-compute all Gaussian distributions in a raster pattern for multiple layers
gaussians = []

rows = 6  # Number of rows
cols = 16  # Number of columns

for layer in range(num_layers):
    for i in range(rows):
        for j in range(cols):
            col = j if i % 2 == 0 else cols - 1 - j  # Raster pattern
            mean = [2 * col, 2 * i]
            bivariate_gauss = multivariate_normal(mean=mean, cov=covariance_matrix)
            Z = bivariate_gauss.pdf(points) * scale
            gaussians.append((Z, layer))

# Animation function called sequentially
def animate(frame):
    ax.clear()
    cs_deposit = np.zeros_like(X)
    if frame < len(gaussians):
        for Z, layer in gaussians[:frame + 1]:
            cs_deposit += Z * (layer + 1)  # Incremental addition for each layer
    
    ax.plot_surface(X, Y, cs_deposit, cmap='plasma')
    ax.set_zlim(0, 5 * num_layers)
    return ax,

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=len(gaussians), interval=25, blit=False)

plt.show()
