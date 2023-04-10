"""Particle filter implementation"""
__all__ = ['ParticleFilterFactory']
from numpy import zeros
from numpy.random import choice


class ParticleFilterFactory:
    # region
    """Particle Filter implementation for autonomous systems, as described in
    "Probabilistic Robotics" by Sebastian Thrun. Provides indirect support for
    nonautonomous systems.

    Directly supports:
        1. vectorized callables (pxt,pzt)
        2. return-by-reference callables (g, h, matrices)
           - if used, ALL callables must return by reference
           - return-by-reference MUST be through 3rd argument
        3. additional (user-supplied) parameters passed to callables

    Indirectly supports:
        1. Nonautonomous systems via direct attribute access (see Examples)
        2. Changing call signatures via subclassing (see Notes)

    REQUIRED INPUTS:
        pxt -- callable -- (X0,u,z)->X1 samples current state given prior state
        pzt -- callable -- (X1,z)->w returns probability of observations zt given state xt

    OPTIONAL INPUTS:
        vec -- bool, optional, default:False -- enable vectorized pxt,pzt calls
        pxt_pars -- iterable, optional, default:[] -- additonal args for pxt
        pzt_pars -- iterable, optional, default:[] -- additonal args for pzt
        P -- int, optional -- required for `rbr`; number of particles in filter
        N -- int, optional -- required for `rbr`; state space size
        rbr -- bool, optional, default:False -- enable return-by-reference pxt,pzt calls
        dbg -- bool, optional, default:False -- log prediction and resampling arrays

    USEFUL ATTRIBUTES:
        pxt_pars/pzt_pars -- Arguments passed to pxt (pzt). Change this to
                             change the extra parameters passed to pxt (pzt).
        Xbart_dbg -- List of predicted particles before resampling. Only
                     populated if `dbg=True`. List is in chronological order
                     (it is appended to during the particle filter call)
        w_dbg -- List of predicted particle probabilities before resampling.
                 Only populated if `dbg=True`. List is in chronological order
                 (it is appended to during the particle filter call)
        ii_dbg -- List of the resampling indices that turned Xbar into X. Only
                  populated if `dbg=True`. List is in chronological order
                  (it is appended to during the particle filter call)

    EXAMPLES:
        Run a single particle filter step:
        >>> pf = ParticleFilterFactory(pxt, pzt)
        >>> Xt1 = pf(Xt0, ut0, zt0)

        Run particle filter when prediction depends on timestep:
        >>> # def pxt(Xtm1, u, dt)
        >>> kwargs = {'pxt_pars': dt}
        >>> pf = ParticleFilterFactory(pxt, pzt, **kwargs)

        Run particle filter with return-by-reference and extra args:
        - return-by-reference must be 3rd argument
        >>> # def pxt(Xtm1, u, Xt, *pxt_pars)
        >>> # def pzt(zt, Xt, wt, *pzt_pars)
        >>> kwargs = {'pxt_pars': dt, ...}
        >>> pf = ParticleFilterFactory(pxt, pzt, **kwargs)

        Run particle filter with non-autonomous transition function:
        >>> # -- initialize --
        >>> # def pxt(Xtm1, u, t)
        >>> kwargs = {'pxt_pars': 0}  # init a time 0
        >>> pf = ParticleFilterFactory(pxt, pzt, **kwargs)
        >>>
        >>> # -- run filter --
        >>> # t = arange(0,tf,dt)
        >>> for i, ti in enumerate(t):
        >>>     Xt[i] = pf(Xt[i-1], u[i], z[i])
        >>>     pf.pxt_pars = [ti]  # arg update -- ENSURE iterable

    NOTES:
        pzt:
            The internal resampling methods normalizes the probabilities by
            necessity (numpy.random.choice), so pzt doesn't need to integrate
            to 1.

        Return-by-Reference Function Calls:
            THIRD (3rd) function arg must be return-by-reference variable.
            Also, ALL callables must be return by reference if this option is
            used. Since it is not possible to tell if a function returns by
            reference, I did not provide an rbr flag for each possible
            callable.

        Return-by-Reference for gGining Speed:
            Currently impossible to fully avoid runtime memory allocation.
            Also, probably only worth it if your arrays are MASSIVE.
            The internal resampling method forces at least one array copy,
            since we're indexing an array with a bunch of indicies.

        Return-by-Reference and Non-vectorized callables:
            Observation probability functions will probably behave weirdly
            if you return-by-reference but did NOT vectorize. This a function
            of the math: pzt returns one number per particle, so when
            iterating over the particles, each pzt call returns a single
            number. Single numbers don't return-by-reference in python.

    REFERENCES:
        Thrun, Probabilistic Robotics, Chp 4.3.
        Thrun, Probabilistic Robotics, Table 4.3.
    """
    # endregion

    def __init__(self, pxt, pzt, **kwargs):
        # basics
        assert callable(pxt), f'pxt must be callable, not {type(pxt)}'
        assert callable(pzt), f'pzt must be callable, not {type(pzt)}'
        self.pxt = pxt
        self.pzt = pzt
        self.pxt_pars = list(kwargs.get('pxt_pars', []))
        self.pzt_pars = list(kwargs.get('pzt_pars', []))

        # debugging
        self.dbg = kwargs.get('debug', False)
        self.Xbart_dbg = []
        self.ii_dbg = []
        self.wt_dbg = []

        # configuration
        vec = kwargs.get('vec', False)
        rbr = kwargs.get('rbr', False)
        P = kwargs.get('P', None)
        N = kwargs.get('N', None)
        self.out = None
        if (P is not None) and (N is not None):
            self.out = zeros((P, N + 1))
        elif rbr:
            raise RuntimeError(
                f'Filter size incomplete:P={P},N={N} and rbr True')
        self._flow = self._flow_factory(vec, rbr)

    def __call__(self, Xtm1, ut, zt):
        """Run particle filter.
        INPUTS:
            Xtm1 -- PxN -- P particles of length-N state vectors at time t-1
            ut -- input at time t
            zt -- K -- observations at time t
            Xt -- optional, PxN -- return-by-reference output
        OUTPUTS:
            Xt -- PxN -- predicted particles at time t
        """
        out = self._flow(Xtm1, ut, zt)
        if self.dbg:
            self.Xbart_dbg.append(out.copy())
        return self._resample(out[:, :-1], out[:, -1])

    def _resample(self, Xbart, wt):
        """resampling step"""
        P = len(Xbart)
        ii = tuple(choice(range(P), size=P, p=wt / wt.sum()))
        # indexing with tuples is sometimes better for memory
        # - see the numpy indexing doc
        if self.dbg:
            self.ii_dbg.append(ii)
            self.wt_dbg.append(wt.copy())
        return Xbart[ii, :]

    # ========================================================================
    # Prediction Step Factory
    #
    # We can iterate, vectorize, and return by reference
    #

    def _flow_factory(self, vec, rbr):
        if vec and rbr:
            return self._flow_vec_rbr
        elif vec:
            return self._flow_vec
        elif rbr:
            return self._flow_iter_rbr
        return self._flow_iter

    def _flow_iter(self, Xtm1, ut, zt):
        """Iterative prediction calculation. Returns directly."""
        out = zeros((Xtm1.shape[0], Xtm1.shape[1] + 1))
        for i in range(len(out)):
            out[i, :-1] = self.pxt(Xtm1[i], ut, *self.pxt_pars)
            out[i, -1] = self.pzt(zt, out[i, :-1], *self.pzt_pars)
        return out

    def _flow_iter_rbr(self, Xtm1, ut, zt):
        """Iterative prediction calculation. Returns-by-reference."""
        self.out[...] = 0
        for i in range(len(self.out)):
            self.pxt(Xtm1[i], ut, self.out[i, :-1], *self.pxt_pars)
            self.out[i, -1] = self.pzt(zt, self.out[i, :-1],
                                       self.out[i, -1], *self.pzt_pars)
        return self.out

    def _flow_vec(self, Xtm1, ut, zt):
        """Vectorized prediction calculation. Returns directly."""
        out = zeros((Xtm1.shape[0], Xtm1.shape[1] + 1))
        out[:, :-1] = self.pxt(Xtm1, ut, *self.pxt_pars)
        out[:, -1] = self.pzt(zt, out[:, :-1], *self.pzt_pars)
        return out

    def _flow_vec_rbr(self, Xtm1, ut, zt):
        """Vectorized prediction calculation. Returns-by-reference."""
        self.out[...] = 0
        self.pxt(Xtm1, ut, self.out[:, :-1], *self.pxt_pars)
        self.pzt(zt, self.out[:, :-1], self.out[:, -1], *self.pzt_pars)
        return self.out
