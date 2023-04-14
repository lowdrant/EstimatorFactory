"""KF implementation"""
__all__ = ['KFFactory']
from warnings import warn

from numpy import asarray, eye, zeros
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


def matmuls(mu, sigma, u, z, A, B, C, R, Q, mu_t=None, sigma_t=None):
    """Generic EKF matmuls"""
    N = mu.size
    mubar = A @ mu + B @ u
    sigmabar = A @ sigma @ A.T + R
    K = sigmabar @ C.T @ pinv(C @ sigmabar @ C.T + Q)
    mu_t = mubar + K @ (z - C @ mubar)
    sigma_t = (eye(N) - K @ C) @ sigmabar
    return mu_t, sigma_t


def matmuls_rbr(mu, sigma, u, z, A, B, C, R, Q, mu_t, sigma_t):
    """Return-by-reference EKF matmuls"""
    N = mu.size
    mubar = A @ mu + B @ u
    sigmabar = A @ sigma @ A.T + R
    K = sigmabar @ C.T @ pinv(C @ sigmabar @ C.T + Q)
    mu_t[...] = mubar + K @ (z - C @ mubar)
    sigma_t[...] = (eye(N) - K @ C) @ sigmabar
    return mu_t, sigma_t


try:
    from numba import njit
    @njit
    def matmuls_njit(mu, sigma, u, z, A, B, C, R, Q, mu_t=None, sigma_t=None):
        """Generic EKF matmuls with njit decorator"""
        N = mu.size
        mubar = A @ mu + B @ u
        sigmabar = A @ sigma @ A.T + R
        K = sigmabar @ C.T @ pinv(C @ sigmabar @ C.T + Q)
        mu_t = mubar + K @ (z - C @ mubar)
        sigma_t = (eye(N) - K @ C) @ sigmabar
        return mu_t, sigma_t

    @njit
    def matmuls_rbr_njit(mu, sigma, u, z, A, B, C, R, Q, mu_t, sigma_t):
        """Return-by-reference EKF matmuls with njit decorator"""
        N = mu.size
        mubar = A @ mu + B @ u
        sigmabar = A @ sigma @ A.T + R
        K = sigmabar @ C.T @ pinv(C @ sigmabar @ C.T + Q)
        mu_t[...] = mubar + K @ (z - C @ mubar)
        sigma_t[...] = (eye(N) - K @ C) @ sigmabar
        return mu_t, sigma_t
except ModuleNotFoundError:
    warn('Supressing njit-optimized functions (Numba module not found).')

    def matmuls_njit(mu, sigma, u, z, G, H, R, Q, mu_t=None, sigma_t=None):
        raise ModuleNotFoundError('Numba not installed')

    def matmuls_rbr_njit(mu, sigma, u, z, G, H, R, Q, mu_t, sigma_t):
        raise ModuleNotFoundError('Numba not installed')


class KFFactory:
    """Construct Kalman Filter as described in "Probabilistic Robotics" by
    Sebastian Thrun.

    Directly supports:
        1. forced, unforced, autonomous, and nonautonomous systems
        2. mixed constant and callable dynamics, covariance matrices
        3. additional (constant) parameters passed to callables
        4. njit-optimized matrix operations, set by flag in init.
        5. return-by-reference callables
           - if used, ALL callables must return by reference
           - return-by-reference MUST be through LAST argument
        6. inferring matrix sizes for memory preallocation

    REQUIRED INPUTS:
        A -- callable or NxN -- state matrix
        B -- callable or NxM -- input matrix
        C -- callable or KxN -- output matrix
        R -- callable or NxN -- state covariance
        Q -- callable or KxK -- measurement covariance

        Callables are expected to take `time` as the first argument, followed
        by additonal parameters. If return-by-reference, the return-by-ref
        argument should be LAST. See Examples.

        Any additional parameters are passed through the `*_pars` keyword args.
        See Notes or Examples.

    OPTIONAL INPUTS:
        n -- int, optional -- state dimension, default: None
        m -- int, optional -- input dimension, default: None
        k -- int, optional -- measurement dimension, default: None
        rbr -- bool, optional -- use matrix return-by-reference, default: False
        njit -- bool, optional -- use njit optimization, default: False
        A_pars,B_pars,...,Q_pars --

    EXAMPLES:
        Single KF step:
        >>> kf = KFFactory(A,B,C,R,Q)
        >>> mu1, sigma1 = kf(mu0, sigma0, observation0, u=u0, t=t0)

        Expected matrix callable signature:
        >>> def A(t, par1, par2, out=None):
        >>>     out[...] = # calculations
        >>>     return out
        >>> A_pars=[A_par1, A_par2])
        >>> kf = KFFactory(A,B,C,R,Q, A_pars=A_pars)
        >>> kf(mu0, sigma0, obs0, u0, t0)

        Run KF when a matrix updates weirdly:
        >>> mu, sigma = zeros((L,N)), zeros((L,N,N))
        >>> mu[-1], sigma[-1] = mu0, sigma0  # for-loop compatability
        >>> kf = KFFactory(A,B,C,R_0,Q)
        >>> for i in range(L):
        >>>     mu[i], sigma[i] = ekf(mu[i-1], sigma[i-1], observation[i])
        >>>     kf.R = asarray(myWeirdRFunc(i))  # ENSURE ARRAY


    NOTES:
        n,m,k:
            Constructor will attempt to infer matrix size from matrices. This
            will not overwrite n, m, or k if they are specified at construction

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

    def __init__(self, A, B, C, R, Q, **kwargs):
        # Matrix Setup
        self.A, self.B, self.C, self.D, self.R, self.Q = A, B, C, D, R, Q
        for key in ('A', 'B', 'C', 'R', 'Q'):
            attr = getattr(self, key)
            if not callable(attr):
                setattr(self, key, asarray(attr))
            pkey = key + '_pars'
            setattr(self, pkey, kwargs.get(pkey, []))

        # Callable Setup
        njit, self.rbr = kwargs.get('njit', False), kwargs.get('rbr', False)
        self._matmuls = factory_matmuls(rbr, njit)

        # Matrix Preallocation
        n, m, k = self._infer_mtxsz(
            kwargs.get('n', None), kwargs.get('m', None), kwargs.get('k', None)
        )
        if any([(v is None) for v in (n, m, k)]):
            warn(f'Matrix sizes only partially specified: '
                 + f'n={n},m={m},k={k}')
            assert not rbr, 'cannot return-by-ref matrices of unknown size'
        self.A_t, self.B_t, self.C_t = None, None, None
        self.R_t, self.Q_t = None, None
        if (n is not None) and (m is not None) and (k is not None):
            self.A_t, self.B_t = zeros((n, n)), zeros((n, m))
            self.C_t = zeros((k, n))
            self.R_t, self.Q_t = zeros((n, n)), zeros((m, m))

    def __call__(self, mu, sigma, z, u=0, t=0):
        """Run Kalman Filter step. This function does not yet support return
        by reference.
        INPUTS:
            mu -- Nx1 -- estimated mean state
            sigma -- NxN -- estimated state covariance
            z -- Kx1 -- measurement
            u -- Mx1, optional -- input to system. default: 0
            t -- float, optional -- time value. default: 0
        OUTPUTS:
            mu_t, sigma_t -- next mean and covariance state estimates
        """
        A_t = self._mtx_wrapper('A', t)  # mem prealloc should return a view
        B_t = self._mtx_wrapper('B', t)
        C_t = self._mtx_wrapper('C', t)
        R_t = self._mtx_wrapper('R', t)
        Q_t = self._mtx_wrapper('Q', t)
        return self._matmuls(mu, sigma, u, z, A_t, B_t, C_t, R_t, Q_t)

    def _infer_mtxsz(self, n, m, k):
        """Infer matrix sizes. No overwrite if n, m, or k given as numbers.
        INPUTS:
            n -- int -- state size, None if unknown
            m -- int -- input size, None if unknown
            k -- int -- observation size, None if unknown
        OUTPUTS:
            n, m, k -- Inferred space sizes. If number is given in function
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
        if (n is None) and (not callable(self.C)):
            n = len(self.C.T)
        elif n is None:
            for key in ('A', 'B', 'R'):
                attr = getattr(self, key)
                if not callable(attr):
                    n = len(attr)
                    break
        if m is None:
            for key in ('B', 'D'):
                attr = getattr(self, key)
                if not callable(attr):
                    m = len(attr.T)
                    break
        if k is None:
            for key in ('C', 'D', 'Q'):
                attr = getattr(self, key)
                if not callable(attr):
                    k = len(attr)
                    break
        return n, m, k

    def _mtx_wrapper(self, key, t):
        """Get matrix of name <key>. Universal interface for matrix attributes.
        INPUTS:
            key -- str -- matrix attribute name, e.g. 'A'
            t -- float -- time value of time step
        RETURNS:
            matrix
        NOTES:
            Matrix Callable and Return-by-Reference:
                Assumes the return-by-reference variable is the LAST argument
                of the function.
            Matrix Class Attributes:
                Assumes each matrix has 3 associated attributes:
                <key> -- the matrix or matrix callable
                <key>_t -- storage variable for a return-by-reference
                           matrix callable. None if unused.
                <key>_pars -- iterable of additional parameters for the
                              matrix callable, if applicable.
        """
        obj = getattr(self, key)
        tgt = getattr(self, key + '_t')
        pars = getattr(self, key + '_pars')
        if callable(obj) and self.rbr:
            obj(t, *pars, tgt)
            return tgt  # ensure return-by-ref by returning separately
        elif callable(obj):
            return obj(t, *pars)
        elif tgt is None:
            return obj.view(obj.dtype)
        tgt[...] = obj.view(obj.dtype)  # ensure view insertion
        return tgt
