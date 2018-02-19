import numpy as np
from scipy import sparse
from scikits.sparse.cholmod import (cholesky, cholesky_AAt,
                                    analyze, analyze_AAt,
                                    CholmodError)
from contexttimer import Timer

N = 5000  # Sparse matrix of this dimension
entries = 50000  # Use this many entries in our sparse matrix

with Timer() as t:
    X = sparse.rand(N, N, density=entries/(N**2))
    K = (X.T * X).tocsc() + sparse.identity(N)
print('Sparse matrix generation elapsed time: {}'.format(t.elapsed))
for mode in ("simplicial", "supernodal"):
    print('\nMode: {}'.format(mode))

    with Timer() as t:
        L = cholesky(K, mode=mode)
    print('Cholesky factorisation elapsed time: {}'.format(t.elapsed))

    # Full inverse
    with Timer() as t:
        inv_K = L.inv()
    print('Full inverse elapsed time: {}'.format(t.elapsed))

    for form in ("lower", "upper", "full"):
        # Sparse inverse
        with Timer() as t:
            spinv_K = L.spinv(form=form)
        print('{} Sparse inverse elapsed time: {}'.format(form, t.elapsed))
