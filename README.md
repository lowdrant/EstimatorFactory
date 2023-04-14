# EstimatorFactory
State estimator factory classes, implemented in Python. Heavily polymorphic.

I have made these classes as polymorphic as I could while preserving both my
sanity and the readability of my code. I have not implemented smoothers.

These classes support:
(1) the mixing of constant matrices and callables that
return matrices, (2) internal matrix njit optimization,
(3) inferring matrix size from user-supplied arguments, (4) callables that
return by reference, (5) returning estimation outputs by reference,
(6) additional arguments for callables.

Vectorization is supported for the UKF and Particle Filter, since particle
computations are intuitively vectorizable.

## Requirements
* numpy -- needed for matrix multiplication and large arrays
* (optional) numba -- used for the optional matrix multiplication speedups

## Quickstart
This repo is designed to work as a module out-of-the-box. Each class gives
a callable built from the typical mathematical parameters. Consult class
docstrings for expected arguments.
```
$ git clone https://github.com/lowdrant/EstimatorFactory.git
$ python3
>>> from EstimatorFactory import EKFFactory
>>> help(EKFFactory)
>>> ekf = EKFFactory(g,h,G,H,R,Q)  # callable
>>> for i, d in enumerate(data):
>>>     mu, sigma = ekf(prevmu, prevsigma, u, d)
```



## Some Thoughts
### On Mathematical Usage
Correctly using these estimators (or filters) relies heavily upon the user.
The math, more than the code, is where these tools shine. Determining a model
and the associated statistical properties for understanding your data is what
will make or break your experience with these tools. I have simply sought to
implement them in the most readable and flexible way for the types of models
and parameters I could forsee being useful.

### On Reading the Code
This codebase makes heavy use of two software design patterns that might
confuse non-software people: "Factory" and "Wrapper."

Each class is a factory that takes parameters and returns a function-like object
which will perform the filter calculations using the user-supplied parameters
(the model and statistical properties). The code gets slightly harder to trace
since Python treats functions as variables -- the classes often overwrite
their internal function names during construction to make the desired
calculation clear.

The classes also wrap the user-supplied functions to support the many use cases
described at the top of this README. These wrappers allow the class methods to
have a unified access point to, say, the state transition function, regardless
of the user specifications. Without wrapping, there would be tremendous code
duplication and even more custom function overwrites to see.

## References
[1] Thrun, Sebastian, et al. Probabilistic Robotics. MIT Press, 2010. ISBN 10: 0262201623ISBN

## Author
Marion Anderson - [lmanderson42@gmail.com](mailto:lmanderson42@gmail.com)
