import numpy as np
import scipy.sparse as sp


def ilu_p(A: sp.spmatrix, p: int):
    """
    Incomplete-LU factorisation with level-of-fill (ILU(p)).

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Square sparse matrix to be factorised.
    p : int
        Maximum allowable fill level.

    Returns
    -------
    (L, U) : tuple of scipy.sparse.csc_matrix
        L is unit-lower-triangular (diagonal stored as 1.0),
        U is upper-triangular.  Both are returned in CSC format
        for efficient triangular solves.
    """
    n = A.shape[0]

    # Work in LIL for fast incremental updates
    A_lil = A.tolil(copy=True)

    # fill_level[(i,j)] records current level; originals start at level-0
    fill_level = {(i, j): 0 for i, j in zip(A.nonzero()[0], A.nonzero()[1])}

    # ------------------------- Algorithm 10.5 -------------------------
    for i in range(1, n):          # i = 2 ... n  (1-based in book)
        for k in range(i):         # k = 1 .. (i-1)
            if (i, k) in fill_level and fill_level[(i, k)] <= p:
                # a(i,k) := a(i,k)/a(k,k)
                A_lil[i, k] = A_lil[i, k] / A_lil[k, k]

                # a(i,*) := a(i,*) - a(i,k)*a(k,*)
                for j in A_lil.rows[k]:
                    if j > k:      # update only to the right of the pivot
                        A_lil[i, j] -= A_lil[i, k] * A_lil[k, j]

                        # ℓ(i,j) = min( ℓ(i,j), ℓ(i,k) + ℓ(k,j) + 1 )
                        old = fill_level.get((i, j), np.inf)
                        new = fill_level.get((i, k), np.inf) \
                              + fill_level.get((k, j), np.inf) + 1
                        if new < old:
                            fill_level[(i, j)] = new

        # 8) drop fill_level > p
        for j in A_lil.rows[i]:
            if fill_level.get((i, j), np.inf) > p:
                A_lil[i, j] = 0.0
    # ------------------------------------------------------------------

    # Assemble separate L and U factors
    L_lil = sp.lil_matrix((n, n), dtype=np.float64)
    U_lil = sp.lil_matrix((n, n), dtype=np.float64)

    for i in range(n):
        for j in A_lil.rows[i]:
            if j < i:
                L_lil[i, j] = A_lil[i, j]
            elif j == i:
                L_lil[i, j] = 1.0          # unit diagonal
                U_lil[i, j] = A_lil[i, j]
            else:                          # j > i
                U_lil[i, j] = A_lil[i, j]

    return L_lil.tocsc(), U_lil.tocsc()


def compute_levels(A: np.ndarray, p: int) -> np.ndarray:
    """
    Symbolically compute the level-of-fill matrix used in ILU(p).

    Parameters
    ----------
    A : ndarray (square)
        Non-zeros are assumed to be the initial pattern (level 0).
    p : int
        Maximum fill level to preserve.

    Returns
    -------
    lev : ndarray
        lev[i, j] stores the final computed level for position (i, j).
    """
    n = A.shape[0]

    lev = np.full_like(A, np.inf, dtype=float)
    lev[A != 0] = 0.0                  # existing non-zeros at level-0
    pattern = (A != 0).astype(int)     # 1 = present in pattern

    for k in range(n):
        for i in range(k + 1, n):
            if pattern[i, k] == 0:     # only if (i,k) is in the pattern
                continue
            for j in range(k + 1, n):
                cand = lev[i, k] + lev[k, j] + 1
                if cand < lev[i, j]:
                    lev[i, j] = cand
                    if cand <= p:
                        pattern[i, j] = 1   # grow sparsity pattern

    return lev
