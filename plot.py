# Plot spatial deformations on the torus.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from compute import minimize_misiolek_index


def plot_deformation(fourier_coef, misiolek_index, n_grid=100, reverse=True):
    """Plot spatial deformation from its Fourier coefficients.

    Parameters :
    -----------
    fourier_coef : array, shape=[(N+1)*(2*N+1),]
        Vector of Fourier coefficients of the spatial deformation.
    misiolek_index : float
        Value of the Misiolek index at deformation.
    n_grid : int, default is 100
        Size of the (x,y) grid on which to plot the spatial deformation.
    reverse : boolean, default is True
        If True, change the sign of the deformation if it is positive at
        the middle of the grid. This allows to always have the same sign
        for the Misiolek index minimizer, which is defined modulo its sign.
    """
    xx = np.linspace(0., 2 * np.pi, n_grid)
    x_grid, y_grid = np.meshgrid(xx, xx)

    if fourier_coef.ndim == 1:
        fourier_coef = np.expand_dims(fourier_coef, axis=0)
        misiolek_index = np.expand_dims(misiolek_index, axis=0)
    n_def = fourier_coef.shape[0]

    fig = plt.figure(figsize=(int(n_def * 4), 4))
    for i in range(n_def):
        z_grid = compute_deformation(fourier_coef[i], n_grid)
        ind = np.floor(n_grid/2).astype('int')
        if reverse and z_grid[ind, ind] > 0:
            z_grid *= - 1.

        ax = fig.add_subplot(1, n_def, i + 1, projection='3d')
        ax.plot_surface(x_grid, y_grid, z_grid, cmap=cm.coolwarm)
        plt.xticks([0, np.pi, 2*np.pi], ['0', '$\pi$', '2$\pi$'])
        plt.yticks([0, np.pi, 2*np.pi], ['0', '$\pi$', '2$\pi$'])
        ax.set_zticks([-1, 0, 1])
        ax.set_zticklabels(['-1', '0', '1'])
        ax.view_init(elev=20., azim=45)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Misiolek index is {np.around(misiolek_index[i], 1)}')
    plt.show()


def compute_deformation(fourier_coef, n_grid=100):
    """Compute spatial deformation at grid points from its Fourier coefficients.

    Parameters :
    -----------
    fourier_coef : array, shape=[(N+1)*(2*N+1),]
        Vector of Fourier coefficients of the spatial deformation.
    n_grid : int, default is 100
        Size of the (x,y) grid on which to compute the spatial deformation.
    """
    n_coef = fourier_coef.shape[0]
    N = int((-3 + np.sqrt(9 + 8 * (n_coef - 1))) / 4)
    xx = np.linspace(0., 2 * np.pi, n_grid)
    x_grid, y_grid = np.meshgrid(xx, xx)
    j_mat = np.tile(np.arange(N+1), (2*N+1, 1)).T
    k_mat = -N + np.tile(np.arange(2*N+1), (N+1, 1))
    coef_mat = fourier_coef.reshape((N+1, 2*N+1))
    fourier_mat = np.cos(np.einsum('ij,kl->ijkl', j_mat, x_grid) + np.einsum('ij,kl->ijkl', k_mat, y_grid))
    mat = np.einsum('ij,ijkl->ijkl', coef_mat, fourier_mat)
    return np.sum(mat, axis=(0, 1))