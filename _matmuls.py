"""!PRIVATE -- KF/EKF matrix multiplication factory

Between the EKF and the KF, the matrix multiplications are the same, but
the variable names differ. These functions use the EKF convention since I
wrote the EKF class first.
"""
__all__ = ['factory_matmuls']
from warnings import warn

from numpy import eye
from numpy.linalg import pinv


def factory_matmuls(rbr, njit):
    """Get the appropriate matrix multiplication function
    INPUTS:
        rbr -- bool -- true if matrix mults should return by reference
        njit -- bool -- true if matrix mults should be numba compiled
    OUTPUTS:
        callable (sigma, z/y, R, Q, H/C, G/A, mubar, zhat) -> mu, sigma
    """
    if rbr and njit:
        return matmuls_rbr_njit
    elif rbr:
        return matmuls_rbr
    elif njit:
        return matmuls_njit
    return matmuls


def matmuls(sigma, z, R, Q, H, G, mubar, zhat, mu_t=None, sigma_t=None):
    """Generic EKF matmuls"""
    N = mubar.size
    sigmabar = G @ sigma @ G.T + R
    K = sigmabar @ H.T @ pinv(H@sigmabar@H.T + Q)
    mu_t = mubar + K @ (z - zhat)
    sigma_t = (eye(N) - K@H)@sigmabar
    return mu_t, sigma_t


def matmuls_rbr(sigma, z, R, Q, H, G, mubar, zhat, mu_t, sigma_t):
    """Return-by-reference EKF matmuls"""
    N = mubar.size
    sigmabar = G @ sigma @ G.T + R
    K = sigmabar @ H.T @ pinv(H@sigmabar@H.T + Q)
    mu_t[...] = mubar + K @ (z - zhat)
    sigma_t[...] = (eye(N) - K@H)@sigmabar
    return mu_t, sigma_t


try:
    from numba import njit
    @njit
    def matmuls_njit(sigma, z, R, Q, H, G, mubar, zhat, mu_t=None, sigma_t=None):
        """Generic EKF matmuls with njit decorator"""
        N = mubar.size
        sigmabar = G @ sigma @ G.T + R
        K = sigmabar @ H.T @ pinv(H@sigmabar@H.T + Q)
        mu_t = mubar + K @ (z - zhat)
        sigma_t = (eye(N) - K@H)@sigmabar
        return mu_t, sigma_t

    @njit
    def matmuls_rbr_njit(sigma, z, R, Q, H, G, mubar, zhat, mu_t, sigma_t):
        """Return-by-reference EKF matmuls with njit decorator"""
        N = mubar.size
        sigmabar = G @ sigma @ G.T + R
        K = sigmabar @ H.T @ pinv(H@sigmabar@H.T + Q)
        mu_t[...] = mubar + K @ (z - zhat)
        sigma_t[...] = (eye(N) - K@H)@sigmabar
        return mu_t, sigma_t
except ModuleNotFoundError:
    warn('Supressing njit-optimized functions (Numba module not found).')

    def matmuls_njit(sigma, z, R, Q, H, G, mubar, zhat, mu_t=None, sigma_t=None):
        raise NotImplementedError('Numba not installed')

    def matmuls_rbr_njit(sigma, z, R, Q, H, G, mubar, zhat, mu_t, sigma_t):
        raise NotImplementedError('Numba not installed')
