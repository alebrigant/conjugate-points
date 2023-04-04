# Test functions from the compute.py file.

from compute import *


TOL = 1e-6


def _poisson_bracket_direct(point, m, n, fourier_coef):
    """Compute the Poisson bracket at a point of the torus.

    This is the Poisson bracket Phi(x, y) = {Psi, f}(x,y) where
    f is a function with Fourier coefficients fourier_coef,
    Psi(x, y) = - cos(mx) * cos(ny) is the stream function, and
    (x, y) are the coordinates of a point of the torus.

    Parameters:
    ----------
    point: array, shape=[2,]
        Point of the Torus.
    fourier_coef: array, shape=[N+1, 2*N+1]
        Fourier coefficients of the function f.
        Must verify fourier_coef[0, :N] = 0.
    """
    x, y = point
    N = fourier_coef.shape[0] - 1
    j_mat = np.tile(np.arange(N+1), (2*N+1, 1)).T
    k_mat = -N + np.tile(np.arange(2*N+1), (N+1, 1))
    fx = - np.sum(fourier_coef * j_mat * np.sin(j_mat * x + k_mat * y))
    fy = - np.sum(fourier_coef * k_mat * np.sin(j_mat * x + k_mat * y))
    psix = m * np.sin(m * x) * np.cos(n * y)
    psiy = n * np.cos(m * x) * np.sin(n * y)
    return psix * fy - psiy * fx


def _laplacian_direct(point, fourier_coef):
    """Compute the Laplacian of function from its Fourier coefficients.

    Parameters:
    ----------
    point: array, shape=[2,]
        Point of the Torus.
    fourier_coef: array, shape=[N+1, 2*N+1]
        Fourier coefficients of the function f.
        Must verify fourier_coef[0, :N] = 0.
    """
    x, y = point
    N = fourier_coef.shape[0] - 1
    j_mat = np.tile(np.arange(N+1), (2*N+1, 1)).T
    k_mat = -N + np.tile(np.arange(2*N+1), (N+1, 1))
    fxx = - np.sum(fourier_coef * j_mat**2 * np.cos(j_mat * x + k_mat * y))
    fyy = - np.sum(fourier_coef * k_mat**2 * np.cos(j_mat * x + k_mat * y))
    return fxx + fyy


def _from_coef_to_value(point, fourier_coef, Nj, Nk):
    """Compute function at point from its Fourier coefficients.

    Parameters:
    ----------
    point: array, shape=[2,]
        Point of the Torus.
    fourier_coef: array, shape=[(Nj + 1) * (2 * Nk + 1),]
        Fourier coefficients of the function f.
        Must verify fourier_coef[0, :Nk] = 0.
    """
    x, y = point
    j_mat = np.tile(np.arange(Nj + 1), (2*Nk + 1, 1)).T
    k_mat = -Nk + np.tile(np.arange(2*Nk + 1), (Nj + 1, 1))
    fourier_coef_mat = fourier_coef.reshape((Nj + 1, 2 * Nk + 1))
    f_at_xx = np.sum(fourier_coef_mat * np.cos(j_mat * x + k_mat * y))
    return f_at_xx


def test_poisson_bracket(N, tol=TOL):
    """Test that all computations of the poisson bracket give the same answer.
    """
    m = np.random.randint(1, 3)
    n = np.random.randint(3, 5)
    fourier_coef = np.random.rand(N+1, 2*N+1)
    fourier_coef[0, :N] = 0
    fourier_coef_vec = fourier_coef.reshape((N+1) * (2*N+1))
    point = np.random.rand(2)
    result_coef = poisson_bracket_mat(m, n, N, exact=1) @ fourier_coef_vec
    result_at_point = _from_coef_to_value(point, result_coef, N + m, N + n)
    expected = _poisson_bracket_direct(point, m, n, fourier_coef)

    assert np.abs(result_at_point - expected) < tol


def test_laplacian(N, tol=TOL):
    """Test that all computations of the Laplacian give the same answer.
    """
    fourier_coef = np.random.rand(N + 1, 2 * N + 1)
    fourier_coef[0, :N] = 0
    fourier_coef_vec = fourier_coef.reshape((N + 1) * (2 * N + 1))
    point = np.random.rand(2)

    result_coef = laplacian_mat(N) @ fourier_coef_vec
    result_at_point = _from_coef_to_value(point, result_coef, N, N)
    expected = _laplacian_direct(point, fourier_coef)
    assert np.all(np.abs(result_at_point - expected) < tol)


def test_laplacian_inverse(N):
    """Verify that the matrix form is the inverse matrix of the Laplacian.

    One needs to remove N+1 first lines and columns, because they correspond to
    j=0 and k<0 (not in the Fourier decomposition) and j=0, k=0 gives diagonal
    element j**2 + k**2 = 0, which prevents Laplacian_mat from being invertible.
    """
    Lbd_mat = - laplacian_mat(N=N)[N + 1:, N + 1:]
    Lbd_inv_mat = - laplacian_inv_mat(N=N, p=1)[N + 1:, N + 1:]
    assert np.all(Lbd_inv_mat == np.linalg.inv(Lbd_mat))


def test_misiolek_index(N):
    """Test Misiolek index shape and particular case.

    Test the output shape and test known value for m=1, n=2 and
    f(x,y) = 4 cos(x+y) + 1/2 cos(2x+y) + 1/2 cos(2x-y).
    """
    m, n = 1, 2
    size = (N + 1) * (2 * N + 1)
    mi_mat = misiolek_index_mat(m, n, N, p=0)
    result = mi_mat.shape
    expected = (size, size)
    assert result == expected

    fourier_coef = np.zeros((N+1, 2*N+1))
    fourier_coef[0, N + 1] = 4
    fourier_coef[2, N + 1] = 1 / 2
    fourier_coef[2, N - 1] = 1 / 2
    fourier_coef = fourier_coef.reshape(size)
    result = fourier_coef.T @ mi_mat @ fourier_coef
    expected = - 19/4 * np.pi ** 2
    assert result == expected