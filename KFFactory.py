"""KF implementation"""
__all__ = ['KFFactory']
from warnings import warn

from numpy import asarray, zeros

from ._matmuls import *


class KFFactory:
    """Construct Kalman Filter for system with state dimension `n`, input
    dimension `m`, and observation dimension `k`.
    """

    def __init__(self, A, B, C, R, Q, D=None, **kwargs):
        self.A, self.B, self.C, self.D, self.R, self.Q = A, B, C, D, R, Q
        for k in ('A', 'B', 'C', 'D', 'R', 'Q'):
            attr = getattr(self, k)
            if attr is None:
                continue
            elif not callable(attr):
                setattr(self, k, asarray(attr))

        njit, callrbr = kwargs.get('njit', False), kwargs.get('callrbr', False)
        n, m, k = self._infer_mtxsz(
            kwargs.get('n', None), kwargs.get('m', None), kwargs.get('k', None)
        )
        if any([(v is None) for v in (n, m, k)]):
            warn(f'Matrix sizes only partially specified: '
                 + f'n={n},m={m},k={k}')
            assert not rbr, 'cannot return-by-ref matrices of unknown size'

        self.A_t, self.B_t, self.C_t, self.D_t = None, None, None, None
        self.R_t, self.Q_t = None, None
        if (n is not None) and (m is not None) and (k is not None):
            self.A_t, self.B_t = zeros((n, n)), zeros((n, m))
            self.C_t, self.D_t = zeros((k, n)), zeros((k, m))
            self.R_t, self.Q_t = zeros((n, n)), zeros((m, m))
        self._matmuls = factory_matmuls(callrbr, njit)

    def __call__(self, mu, sigma, z, mu_t=None, sigma_t=None):
        raise NotImplementedError
        # return self._matmuls(sigma, z, R, Q, C, B, mubar, zhat, mu_t, sigma_t)

    # ========================================================================
    # Setup

    def _infer_mtxsz(self, n, m, k):
        """Infer matrix size."""
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

    # ========================================================================
    # Dynamics Wrappers

    @staticmethod
    def _mtx_wrapper(obj, tgt, args):
        """Universal logic for matrix functions.
        INPUTS:
            obj -- object with desired data
            tgt -- ndarray where desired data will be stored
            args -- arguments `obj` would take if `obj` is callable
        """
        if not callable(obj):
            if tgt is None:
                return obj.view(obj.dtype)
            tgt[...] = obj.view(obj.dtype)
            return tgt
        return obj(*args)

    def _Afun(self):
        args = [self.A_t] if self.rbr else []
        return self._mtx_wrapper(self.A, self.A_t, args + self.A_pars)

    def _Bfun(self):
        args = [self.B_t] if self.rbr else []
        return self._mtx_wrapper(self.B, self.B_t, args + self.B_pars)

    def _Cfun(self):
        args = [self.C_t] if self.rbr else []
        return self._mtx_wrapper(self.C, self.C_t, args + self.C_pars)

    def _Dfun(self):
        args = [self.D_t] if self.rbr else []
        return self._mtx_wrapper(self.D, self.D_t, args + self.D_pars)

    def _Rfun(self):
        """Wrap R with universal function call"""
        args = [self.R_t] if self.rbr else []
        return self._mtx_wrapper(self.R, self.R_t, args + self.R_pars)

    def _Qfun(self):
        """Wrap Q with universal function call"""
        args = [self.R_t] if self.rbr else []
        return self._mtx_wrapper(self.Q, self.Q_t, args + self.Q_pars)
