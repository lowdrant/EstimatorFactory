"""EKF implementation"""
__all__ = ['EKFFactory']
from warnings import warn

from numpy import asarray, eye, isscalar, zeros
from numpy.linalg import pinv

from .KFFactory import KFFactory


def factory_matmuls(rbr, njit):
    """Get the appropriate matrix multiplication function.
    INPUTS:
        rbr -- bool -- true if matrix mults should return by reference
        njit -- bool -- true if matrix mults should be numba compiled
    OUTPUTS:
        callable -- signature:
            (mu, sigma, u, z, A, B, C, R, Q, mu_t=None, sigma_t=None)
            ->
            (mu_t, sigma_t)
    NOTES:
        Why so many matmul arguments:
            For the most mileagle from njit and return-by-ref settings,
            it is good to (1) minimize the number of numpy function calls,
            e.g. `H(...)`, (2) minimize the number of intermediate arrays.
    """
    if rbr and njit:
        return matmuls_rbr_njit
    elif rbr:
        return matmuls_rbr
    elif njit:
        return matmuls_njit
    return matmuls


def matmuls(mubar, sigma, z, zhat, G, H, R, Q, mu_t=None, sigma_t=None):
    """Generic EKF matmuls"""
    N = mubar.size
    sigmabar = G @ sigma @ G.T + R
    K = sigmabar @ H.T @ pinv(H@sigmabar@H.T + Q)
    mu_t = mubar + K @ (z - zhat)
    sigma_t = (eye(N) - K@H)@sigmabar
    return mu_t, sigma_t


def matmuls_rbr(mubar, sigma, z, zhat, G, H, R, Q, mu_t, sigma_t):
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
    def matmuls_njit(mubar, sigma, z, zhat, G, H, R, Q, mu_t=None, sigma_t=None):
        """Generic EKF matmuls with njit decorator"""
        N = mubar.size
        sigmabar = G @ sigma @ G.T + R
        K = sigmabar @ H.T @ pinv(H@sigmabar@H.T + Q)
        mu_t = mubar + K @ (z - zhat)
        sigma_t = (eye(N) - K@H)@sigmabar
        return mu_t, sigma_t

    @njit
    def matmuls_rbr_njit(mubar, sigma, z, zhat, G, H, R, Q, mu_t, sigma_t):
        """Return-by-reference EKF matmuls with njit decorator"""
        N = mubar.size
        sigmabar = G @ sigma @ G.T + R
        K = sigmabar @ H.T @ pinv(H@sigmabar@H.T + Q)
        mu_t[...] = mubar + K @ (z - zhat)
        sigma_t[...] = (eye(N) - K@H)@sigmabar
        return mu_t, sigma_t
except ModuleNotFoundError:
    warn('Supressing njit-optimized functions (Numba module not found).')

    def matmuls_njit(mubar, sigma, z, zhat, G, H, R, Q, mu_t=None, sigma_t=None):
        """matmuls_njit placeholder for when Numba not present"""
        raise ModuleNotFoundError('Numba not installed')

    def matmuls_rbr_njit(mubar, sigma, z, zhat, G, H, R, Q, mu_t, sigma_t):
        """matmuls_rbr_njit placeholder for when Numba not present"""
        raise ModuleNotFoundError('Numba not installed')


class EKFFactory(KFFactory):
    # region
    """EKF implementation for autonomous systems, as described in
    "Probabilistic Robotics" by Sebastian Thrun.

    Directly supports:
        1. forced, unforced, autonomous, and nonautonomous systems
        2. mixed constant and callable dynamics, covariance matrices
        3. additional (constant) parameters passed to callables
        4. njit-optimized matrix operations, set by flag in init.
        5. return-by-reference callables
           - if used, ALL callables must return by reference
           - return-by-reference MUST be through LAST argument
        6. inferring matrix sizes for memory preallocation

    Indirectly supports:
        1. unorthodox call signatures via direct attribute access

    REQUIRED INPUTS:
        g -- callable -- state transition function; (t,u,mu)->mubar
        h -- callable -- measurement function; (t,mubar)->zhat
        G -- callable or NxN -- state transition Jacobian; (t,u,mu)->NxN
        H -- callable or MxN -- measurement Jacobian; (t,mubar)->MxN
        R -- callable or NxN -- state covariance; (t)->NxN
        Q -- callable or MxM -- measurement covariance; (t)->MxM

        Callables are expected to take `time` as the first argument, followed
        by additonal parameters. If return-by-reference, the return-by-ref
        argument should be LAST. See Examples. For autonomous or unforced
        systems, simply have the callable not use the time or system input
        variable.

        Callables can also be passed additional (constant) parameters by
        means of the `*_pars` optional keyword args. See Notes.

    OPTIONAL INPUTS:
        n -- int, default:None -- state dimension
        k -- int, default:None -- measurement dimension
        rbr -- bool, default:False -- set true if callables return by reference
        callrbr -- bool, default:False -- set true if EKF call should return by reference
        njit -- bool, default:False -- use njit optimization for matrix operations
        g_pars,h_pars,...,Q_pars -- parameters for callables. See Notes

    EXAMPLES:
        Run a single EKF step:
        >>> ekf = EKF(g,h,G,H,R,Q)
        >>> mu1, sigma1 = ekf(mu0, sigma0, observation0, u0, t0)

        Configure to return estimates by reference:
        >>> mu1, sigma1 = zeros_like(mu0), zeros_like(sigma0)
        >>> ekf = EKF(g,h,G,H,R,Q, callrbr=True)
        >>> ekf(mu0, sigma0, observation0, mu1, sigma1)

        Run EKF when a dynamics matrix updates weirdly:
        >>> mu, sigma = zeros((L,N)), zeros((L,N,N))
        >>> mu[-1], sigma[-1] = mu0, sigma0  # for loop compatability
        >>> ekf = EKF(g,h,G,H,R,Q)
        >>> for i in range(L):
        >>>     mu[i], sigma[i] = ekf(mu[i-1], sigma[i-1], observation[i], ...)
        >>>     ekf.R = asarray(myWeirdRFunc(i))  # ENSURE ARRAY

    NOTES:
        n,k:
            Constructor will attempt to infer matrix size from matrices. This
            will not overwrite n or k if they are specified at construction

        return-by-reference:
            LAST function arg must be return-by-reference variable. Also, ALL
            callables must be return by reference if this option is used.
            Since it is not possible to tell if a function returns by
            reference, I did not provide an rbr flag for each possible
            callable.

        Callables with more args:
            Each callable also has an assosciated keyword argument,
            <callable_name>_pars, which is an iterable e of additional
            parameters to be passed when called. It will be passed like so
            for `A`: `A(t, *A_pars, self.A_t)`.

        Callables with very different call signatures:
            Subclass and overwrite the relevant wrappers.

    REFERENCES:
        Thrun, Probabilistic Robotics, Chp 3.3.
        Thrun, Probabilistic Robotics, Table 3.3.
    """
    # endregion

    def __init__(self, g, h, G, H, R, Q, **kwargs):
        # EKF Setup
        self.g, self.h = g, h
        self.G = G if callable(G) else asarray(G)
        self.H = H if callable(H) else asarray(H)
        self.R = R if callable(R) else asarray(R)
        self.Q = Q if callable(Q) else asarray(Q)
        for key in ('g', 'h', 'G', 'H', 'R', 'Q'):  # k is a variable later
            pkey = key + '_pars'
            pars = kwargs.get(key + '_pars', [])
            setattr(self, pkey, pars)

        # Size Inference and Preallocation
        n, k = self._infer_mtxsz(kwargs.get('n', None), kwargs.get('k', None))
        for key in ('mubar', 'zhat', 'G_t', 'H_t', 'R_t', 'Q_t'):
            setattr(self, key, None)
        if (n is None) or (k is None):
            warn('Unable to infer matrix size. Return be reference will fail')
        else:
            self.mubar, self.zhat = zeros(n), zeros(k)
            self.G_t, self.H_t = zeros((n, n)), zeros((k, n))
            self.R_t, self.Q_t = zeros((n, n)), zeros((k, k))

        # Implementation Selection
        self.rbr = kwargs.get('rbr', False)
        njit = kwargs.get('njit', False)
        callrbr = kwargs.get('callrbr', False)
        self._matmuls = factory_matmuls(callrbr, njit)
        self._init_safety_checks(n, k)

    def __call__(self, mu, sigma, z, u=0, t=0, mu_t=None, sigma_t=None):
        """run EKF - see Thrun, Probabilistic Robotics, Table 3.3 """
        mubar = self._gfun(t, u, mu)
        zhat = self._hfun(t, mubar)
        G = self._mtx_wrapper('G', t, (u, mu))  # should return by ref
        H = self._mtx_wrapper('H', t, (mubar,))
        R = self._mtx_wrapper('R', t)
        Q = self._mtx_wrapper('Q', t)
        return self._matmuls(mubar, sigma, z, zhat, G, H, R, Q, mu_t, sigma_t)

    def _infer_mtxsz(self, n, k):
        """Infer matrix sizes. No overwrite if n, k given as numbers.
        INPUTS:
            n -- int -- state size, None if unknown
            k -- int -- observation size, None if unknown
        OUTPUTS:
            n, k -- Inferred space sizes. If number is given in function
                    call, same number will be returned. Otherwise, size
                    of relevant matrix. If unable to infer, None.
        NOTES:
            No Error Checks:
                Does not check for matrix size consistency, or that matrix size
                matches the provided numbers.
            No Overwrite:
                If a number is given for, e.g. `n`, `n` will be returned with
                that same value.
        """
        if (n is None) and (not callable(self.H)) and (self.H.ndim > 0):
            n = len(self.H.T)
        elif self.H.ndim == 0:
            n = 1
        elif n is None:
            for key in ('G', 'R'):
                n = self._infer_mtxsz_rows(key)
        if k is None:
            for key in ('H', 'Q'):
                k = self._infer_mtxsz_rows(key)
        return n, k

    def _init_safety_checks(self, n, k):
        """Provide informative error messages to user."""
        assert callable(self.g), 'g must be callable'
        assert callable(self.h), 'h must be callable'
        if (n is None) ^ (k is None):
            warn(f'Matrix sizes only partially specified: n={n},k={k}')
        if (n is None) or (k is None):
            assert not self.rbr, 'cannot return-by-ref matrices of unknown size'

    def _gfun(self, t, u, mu):
        """g wrapper"""
        if self.rbr:
            return self.g(t, u, mu, *self.g_pars, self.mubar)
        return self.g(t, u, mu, *self.g_pars)

    def _hfun(self, t, mubar):
        """h wrapper"""
        if self.rbr:
            return self.h(t, mubar, *self.h_pars, self.zhat)
        return self.h(t, mubar, *self.h_pars)
