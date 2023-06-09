# Find conjugate points along Kolmogorov flows on the torus by minimizing the Misiolek criterion.

import numpy as np


def minimize_misiolek_index(m, n, N, p=0, constrained=False):
    """Minimize the Misiolek index on spatial deformations of unit p-Sobolev norm.

    Find the vector field along the geodesic generated by the stream function
    psi(x,y) = cos(mx)cos(ny) in the space of volume-preserving diffeomorphisms of the torus,
    that minimizes the Misiolek index. We restrict to vector fields generated by a spatial
    deformation f(x,y) in the following way: Y = sgrad g, where g(t,x,y) = sin(t/T)f(x,y),
    and search for minimizers f of unit p-homogenous Sobolev norm. The Misiolek index can be
    represented by a quadratic operator acting on the spatial deformation f via the homogeneous
    Sobolev inner product of order p. The minimal value of the operator on deformations of unit
    norm is given by the minimal eigenvalue of its matrix representation, and the associated
    eigenvector gives the minimizer f. When the eigenvalue is negative, then the index form
    is negative for some sufficiently large time and there is a conjugate point along the
    geodesic for the considered parameters m and n.

    The matrix representation is computed with respect to the coordinate system defined
    by the Fourier coefficients of f = sum_{j, k} a_{jk} cos(jx+ky), where the sum is
    for j = 0 and k = 0 ... N, and j = 1 ... N and k = -N ... N. Each spatial function f
    is represented by the vector of its Fourier coefficients x_vec of shape (N+1) * (2*N+1),
    such that if x = x_vec.reshape(N+1, 2*N+1), then
    x[0, 0:N] = 0, x[0, N:2*N+1] = a_{jk} for k = 0 ... N
    x[1:N+1, 0:2*N+1] = a_{jk} for j = 1 ... N and k = -N ... N.

    Parameters :
    -----------
    m, n : int
        Integers that define the stream function psi(x,y) = - cos(mx) cos(ny) generating the
        geodesic in the space of volume-preserving diffeomorphisms. psi is an eigenfunction
        of the Laplacian associated to the eigenvalue -(m**2 + n**2).
    N : int
        Maximum order of the Fourier coefficients of the spatial deformation f.
    p : int
        Order of the homogeneous Sobolev inner product with respect to which the matrix
        representation of the Misiolek index is computed.
    constrained : boolean, default is False
        If True, restrict to solutions with coefficient a_{01} = 0. This allows to eliminate
        one of the two minimizers in the case m=n, to always obtain a perturbation of cos(x)
        and not a perturbation of cos(y).

    Returns :
    ---------
    min_deformation : array, shape=[(N+1) * (2*N+1),]
        Eigenvector corresponding to minimal eigenvalue. This contains the Fourier coefficients
        of the spatial deformation that minimizes the Misiolek index.
    min_value : float
        Minimal value of the Misiolek index.
    """
    mi_mat = misiolek_index_mat(m, n, N, p)
    if constrained:
        mi_mat = np.delete(mi_mat, N + 1, axis=0)
        mi_mat = np.delete(mi_mat, N + 1, axis=1)
    eigenval, eigenvec = np.linalg.eig(mi_mat)
    ind = np.argmin(np.real(eigenval))
    min_value = np.real(eigenval[ind])
    min_deformation = np.real(eigenvec[:, ind])
    if constrained:
        min_deformation = np.hstack((min_deformation[:N + 1], np.zeros(1), min_deformation[N + 1:]))
    return min_deformation, min_value


def misiolek_index_mat(m, n, N, p=0):
    """Compute matrix representation of the Misiolek index.

    Compute the matrix representation of the Misiolek index with respect to the
    homogeneous Sobolev norm or order p. This is:
    \Omega = - 2 * \Pi**2 * \Lambda^{-p} @ L @ (\Lambda - (m**2 + n**2) I) @ L
    where \Lambda is minus Laplacian and L is the Poisson bracket with the stream
    function psi(x,y) = - cos(mx) cos(ny). The Misiolek index for deformation f
    is <f, Omega f>_p where <.,.>_p is the \dot H^p norm.

    Parameters :
    -----------
    m, n : int
        Integers that define the stream function psi(x,y) = - cos(mx) cos(ny).
    N : int
        Maximum order of the Fourier coefficients of f.

    Returns :
    ---------
    mi_mat: array, shape=[(N+1)*(2*N+1), (N+1)*(2*N+1)]
        Matrix representation of the Misiolek index.
    """
    lbd = m**2 + n**2
    size = (N+1) * (2*N+1)
    pb_mat = poisson_bracket_mat(m, n, N)
    lbd_mat = - laplacian_mat(N)
    lbd_inv_mat = (-1)**p * laplacian_inv_mat(N, p)
    mi_mat = - lbd_inv_mat @ pb_mat @ (lbd_mat - lbd * np.eye(size)) @ pb_mat
    return 2 * np.pi**2 * mi_mat


def poisson_bracket_mat(m, n, N, exact=0):
    """Compute matrix form of the poisson bracket operator.

    This is Lf = {f, psi} where psi(x,y) = - cos(mx) cos(ny) is the stream function.

    Parameters :
    -----------
    m, n : int
        Integers that define the stream function psi(x,y) = - cos(mx) cos(ny).
    N : int
        Maximum order of the Fourier coefficients of f.
    exact : boolean, default is 0.
        Indicates whether the rectangle matrix representation of the Poisson bracket
        operator should be truncated to a square matrix (exact = 0) or not (exact = 1).

    Returns :
    ---------
    mat: array
        Matrix representation of the Poisson bracket linear operator, i.e. if x_vec contains
        the Fourier coefficients of f, than L(f) = mat @ x_vec.
        If exact = 0, shape=[(N+1)*(2*N+1), (N+1)*(2*N+1)]
        If exact = 1, shape=[(N+m+1)*(2*N+2*n+1), (N+1)*(2*N+1)]
    """
    if N < np.max((m, n)):
        raise ValueError(
            'The number of Fourier coefficients N '
            'must be larger than m and n.')
    size = (N + 1) * (2*N + 1)
    size1 = (N + exact*m + 1) * (2*N + exact*2*n + 1)
    mat = []
    # j = 0
    for k in range(-N - exact*n, N + exact*n + 1):
        mat_0k = np.zeros((N+1, 2*N+1))
        if k >= 0 and n-k+N >= 0:
            mat_0k[m, n-k+N] += m*k
        if k >= 0 and n+k+N < 2*N+1:
            mat_0k[m, n+k+N] -= m*k
        if k >= 0 and -k-n+N >= 0:
            mat_0k[m, -k-n+N] += m*k
        if k >= 0 and k-n+N >= 0:
            mat_0k[m, k-n+N] -= m*k
        mat.append(mat_0k)
    # 0 < j < m
    for j in range(1, m):
        for k in range(-N - exact*n, N + exact*n + 1):
            mat_jk = np.zeros((N+1, 2*N+1))
            if 0 <= n-k+N < 2*N+1:
                mat_jk[m-j, n-k+N] += (m*k - n*j)
            if m+j < N+1 and n+k+N < 2*N+1:
                mat_jk[m+j, n+k+N] += (n*j - m*k)
            if -k-n+N >= 0:
                mat_jk[m-j, -k-n+N] += (m*k + n*j)
            if m+j < N+1 and k-n+N >= 0:
                mat_jk[m+j, k-n+N] += (- m*k - n*j)
            mat.append(mat_jk)
    # j = m
    for k in range(-N - exact*n, N + exact*n + 1):
        mat_mk = np.zeros((N+1, 2*N+1))
        if np.abs(k-n)+N < 2*N+1:
            mat_mk[0, np.abs(k-n)+N] += (m*k - n*m)
        if 2*m < N+1 and k+n+N < 2*N+1:
            mat_mk[2*m, k+n+N] += (n*m - m*k)
        if np.abs(k+n)+N < 2*N+1:
            mat_mk[0, np.abs(k+n)+N] += (m*k + n*m)
        if 2*m < N+1 and k-n+N >= 0:
            mat_mk[2*m, k-n+N] += (- m*k - n*m)
        mat.append(mat_mk)
    # m < j < N+1
    for j in range(m+1, N + exact*m + 1):
        for k in range(-N - exact*n, N + exact*n + 1):
            mat_jk = np.zeros((N+1, 2*N+1))
            if k-n+N >= 0:
                mat_jk[j-m, k-n+N] += (m*k - n*j)
            if j+m < N+1 and k+n+N < 2*N+1:
                mat_jk[j+m, k+n+N] += (n*j - m*k)
            if k+n+N < 2*N+1:
                mat_jk[j-m, k+n+N] += (m*k + n*j)
            if j+m < N+1 and k-n+N >= 0:
                mat_jk[j+m, k-n+N] += (- m*k - n*j)
            mat.append(mat_jk)
    return 1/4 * np.stack(mat).reshape((size1, size))


def laplacian_mat(N):
    """Compute matrix representation of the Laplacian operator.

    Parameters :
    -----------
    N : int
        Maximum order of the Fourier coefficients of f.

    Returns :
    ---------
    mat: array, shape=[(N+1)*(2*N+1), (N+1)*(2*N+1)]
        Matrix representation of the Laplacian linear operator, i.e. if x_vec contains
        the Fourier coefficients of f, than \Delta(f) = mat @ x_vec.
    """
    size = (N+1) * (2*N+1)
    mat = []
    for k in range(-N, N+1):
        mat_0k = np.zeros((N+1, 2*N+1))
        if k >= 0:
            mat_0k[0, k+N] = - k**2
        mat.append(mat_0k)
    for j in range(1, N+1):
        for k in range(-N, N+1):
            mat_jk = np.zeros((N+1, 2*N+1))
            mat_jk[j, k+N] = - (j**2 + k**2)
            mat.append(mat_jk)
    return np.stack(mat).reshape((size, size))


def laplacian_inv_mat(N, p):
    """Compute matrix representation of the p-inverse Laplacian.

    This is \Delta^(-p} where \Delta is the Laplacian operator.

    Parameters :
    -----------
    N : int
        Maximum order of the Fourier coefficients of f.
    p : int
        Power of the inverse.

    Returns :
    ---------
    mat: array, shape=[(N+1)*(2*N+1), (N+1)*(2*N+1)]
        Matrix representation of the inverse Laplacian linear operator, i.e. if x_vec contains
        the Fourier coefficients of f, than \Delta^(-p}(f) = mat @ x_vec.
    """
    size = (N + 1) * (2 * N + 1)
    mat = []
    for k in range(-N, N + 1):
        mat_0k = np.zeros((N + 1, 2 * N + 1))
        if k > 0:
            mat_0k[0, k + N] = (-1)**p * (k ** 2)**(-p)
        mat.append(mat_0k)
    for j in range(1, N + 1):
        for k in range(-N, N + 1):
            mat_jk = np.zeros((N + 1, 2 * N + 1))
            mat_jk[j, k + N] = (-1)**p * (j ** 2 + k ** 2)**(-p)
            mat.append(mat_jk)
    return np.stack(mat).reshape((size, size))