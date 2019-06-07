import enum
import itertools
from collections import namedtuple

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import cvxopt

from hiergp.kernels import sqdist


class VarMethod(enum.Enum):
    CONVEX = 0
    EIGEN = 1
    CUTEIG = 2


class MuMethod(enum.Enum):
    LINEAR = 0
    ALLCOMP = 1
    BESTCOMP = 2


ModelFactors = namedtuple('ModelFactors',
                          ['ubound',
                           'lbound',
                           'scale',
                           'C',
                           'Ki',
                           'EV_MIN',
                           'L',
                           'X',
                           'pmu'])


def est_bound(soln,
              model_data,
              var_method=VarMethod.EIGEN,
              mu_method=MuMethod.BESTCOMP):
    """
    soln: partial vector
    """
    empty = np.argwhere(np.isnan(soln)).ravel()
    nonempty = np.argwhere(np.invert(np.isnan(soln))).ravel()

    ubound = model_data.ubound
    lbound = model_data.lbound
    scale = model_data.scale
    C = model_data.C
    Ki = model_data.Ki
    EV_MIN = model_data.EV_MIN
    L = model_data.L
    X = model_data.X
    pmu = model_data.pmu

    # Estimate the Max. Variance
    if var_method == VarMethod.CONVEX:
        # Estimating variance
        kl = np.exp(-ubound)
        kh = np.exp(-lbound)
        km = (kh-kl)/2

        B_lnk = (np.log(kh)-np.log(kl))/(kh-kl)
        # Derivative at the
        B_neg_lnk = -1/km
        A_lnk = np.log(kh) - B_lnk*kh
        A_neg_lnk = np.log(km) - B_neg_lnk

        # Linear approximation of the square factor
        empty_dim = empty.shape[0]
        P = np.zeros((X.shape[0]+empty_dim, X.shape[0]+empty_dim))
        P[:X.shape[0], :X.shape[0]] = 2*Ki
        q = np.zeros(X.shape[0]+empty_dim)

        G = np.vstack(
            (np.eye(X.shape[0]+empty_dim), -np.eye(X.shape[0]+empty_dim)))
        h = np.hstack((kh, np.ones(empty_dim), -kl, np.zeros(empty_dim)))

        # Construct the A and b matrices row by row
        A = np.zeros((X.shape[0], X.shape[0]+empty_dim))
        A2 = np.zeros((X.shape[0], X.shape[0]+empty_dim))
        b = np.zeros(X.shape[0])
        b2 = np.zeros(X.shape[0])
        for k in range(Ki.shape[0]):
            # Linear approximator for quadratic terms
            A[k, k] = B_lnk[k]
            xl = 0
            xh = 1
            xm = 0.5
            const = 0.5 * \
                np.sum((soln[nonempty]-X[k, nonempty])**2/L[nonempty]**2)

            B_quad = 2*(xm-X[k, empty])/L[empty]**2
            A[k, X.shape[0]:] = 0.5*B_quad
            A_quad = (xm-X[k, empty])**2/L[empty]**2-B_quad*xm
            b[k] = -(0.5*np.sum(A_quad)+A_lnk[k]) - const

            A2[k, k] = -B_neg_lnk[k]
            B_quad2 = -((xh-X[k, empty])**2-(xl-X[k, empty])
                        ** 2)/L[empty]**2/(xh-xl)
            A_quad2 = (xl-X[k, empty])**2/L[empty]**2-B_quad2*xl
            b2[k] = (0.5*np.sum(A_quad2)+A_neg_lnk[k]) + const

        GG = np.vstack((G, A, A2))
        hh = np.hstack((h, b, b2))
        cvxopt.solvers.options['show_progress'] = False
        cvxres = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q),
                                   cvxopt.matrix(GG), cvxopt.matrix(hh))
        max_s2 = scale - cvxres['primal objective']*scale**2
        max_std = np.sqrt(max_s2)
    elif var_method == VarMethod.EIGEN:
        kk = scale*np.exp(-ubound)
        max_s2 = scale-np.dot(kk, kk)*EV_MIN
        max_std = np.sqrt(max_s2)
    elif var_method == VarMethod.CUTEIG:
        num_dim_approx = 6
        sorted_L = np.argsort(L)

        # Fix the five more influential dimensions that have not be
        # set already
        # We need to expand this to the YAML descriptors
        set_dimensions = []
        for d in sorted_L:
            if np.isnan(soln[d]):
                set_dimensions.append(d)
            if len(set_dimensions) == num_dim_approx:
                break

        # Generate solution variants
        soln_variant = np.tile(soln, (2**num_dim_approx, 1))
        var_gen = [[0, 1] for _ in range(num_dim_approx)]
        for k, permutation in enumerate(itertools.product(*var_gen)):
            soln_variant[k, set_dimensions] = permutation

        distances = np.empty((2**num_dim_approx, X.shape[0]))
        for k in range(2**num_dim_approx):
            dist = (soln_variant[k, :]-X)**2/L**2
            dist[np.isnan(dist)] = 0
            distances[k] = np.sum(dist, axis=1)

        # Add the remaining dimensions
        for d in range(X.shape[1]):
            if np.isnan(soln[d]) and d not in set_dimensions:
                distances += 1/L[d]**2
        distances = scale**2*np.exp(-distances)
        min_distances = np.sum(distances, axis=1)
        max_s2 = scale-max(min_distances)*EV_MIN
        max_std = np.sqrt(max_s2)

    # Estimate the Mean
    if mu_method == MuMethod.LINEAR:
        mid_bound = (ubound+lbound)/2
        s_C = C*scale
        B = np.where(s_C < 0,
                     s_C*(np.exp(-ubound)-np.exp(-lbound))/(ubound-lbound),
                     -s_C*np.exp(-mid_bound))
        A = np.where(s_C < 0,
                     s_C*np.exp(-ubound) - B*ubound,
                     s_C*np.exp(-mid_bound)*(1+mid_bound))
        B = B.reshape(-1, 1)
        x2coef = 0.5*B.sum()/L**2
        xcoef = B*X

        const = np.sum(A)+0.5*np.sum(np.sum(xcoef*X, axis=0)/L**2)
        xcoef = -np.sum(xcoef, axis=0)/L**2
        const += pmu
        lower_bound = const

        for ii, val in enumerate(soln):
            if np.isnan(val):
                lower_bound += min([x2coef[ii]*d**2 +
                                    xcoef[ii]*d for d in [0, 1]])
            else:
                lower_bound += x2coef[ii]*val**2+xcoef[ii]*val

        # lower bound is my own fast approximation but it's still bad
        muf = pmu + np.where(C < 0, scale*np.exp(-lbound)
                             * C, scale*np.exp(-ubound)*C).sum()
        min_mu = max(muf, lower_bound)
        yield min_mu, max_s2, max_std

    elif mu_method == MuMethod.ALLCOMP or mu_method == MuMethod.BESTCOMP:
        # Define gradient and function for mean minimization
        LLL = L[empty]**2
        XXX = X[:, empty]
        XL = X[:, empty]/L[empty]
        ss = soln[nonempty].reshape((1, -1))
        CS = C*scale*sqdist(ss/L[nonempty], X[:, nonempty]/L[nonempty])

        XL = X/L
        CS = C*scale

        def func(s):
            x = np.empty((1, X.shape[1]))
            x[0, nonempty] = soln[nonempty]
            x[0, empty] = s
            # Compute mean function
            fun = CS*sqdist(x/L, XL)
            jac = -np.einsum('ij,jk->k', fun, (s-XXX))/LLL
            return (np.sum(fun), jac)

        # When we have no solution, skip the bound computation and just
        # continue exploration
        bounds = [(0, 1) for _ in range(len(empty))]
        min_mu = float('inf')
        count = 0

        # Scale factor by computing the conditional gaussian
        proj_dist = C * \
            np.exp(-0.5*np.sum((soln[nonempty] -
                                X[:, nonempty])**2/L[nonempty]**2, axis=1))
        # Now compute the distances
        Xdist = scale*sqdist(X[:, empty]/L[empty], X[:, empty]/L[empty])
        sC = np.sum(proj_dist*Xdist, axis=1)

        for k, c in sorted(enumerate(sC), key=lambda x: x[1], reverse=False):
            c = C[k]

            if c < 0:
                x0 = X[k, empty]
                count += 1

                res = fmin_l_bfgs_b(func, x0, fprime=None, approx_grad=False,
                                    bounds=bounds)
                resf = res[1]

                if resf + pmu < min_mu:
                    min_mu = resf + pmu
                    yield min_mu, max_s2, max_std

                if mu_method == MuMethod.BESTCOMP:
                    break
