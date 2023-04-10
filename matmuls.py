"""!PRIVATE -- storage for KF/EKF matrix multiplications

Between the EKF and the KF, the matrix multiplications are the same, but
the variable names differ. These functions use the EKF convention since I
wrote the EKF class first.
"""
__all__ = ['matmuls' + v for v in ('', '_rbr', '_njit', '_rbr_njit')]
from warnings import warn

from numpy import eye
from numpy.linalg import pinv


def matmuls(sigma, z, R, Q, H, G, mubar, zhat):
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
    def matmuls_njit(sigma, z, R, Q, H, G, mubar, zhat):
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

    def matmuls_njit(sigma, z, R, Q, H, G, mubar, zhat):
        raise NotImplementedError('Numba not installed')

    def matmuls_rbr_njit(sigma, z, R, Q, H, G, mubar, zhat, mu_t, sigma_t):
        raise NotImplementedError('Numba not installed')
