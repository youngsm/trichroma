import numpy as np
import uncertainties
from uncertainties import unumpy

from chroma.rootimport import ROOT

def build_design_matrix(x, y):
    y_invsigma = 1.0/unumpy.std_devs(y)
    dims = x.shape[1]
    n = 1 + dims + dims*(dims+1)/2
    
    A = np.zeros(shape=(len(x),n))

    A[:,0] = 1.0 * y_invsigma
    for i in range(dims):
        A[:,1+i] = x[:,i] * y_invsigma

    col = 1 + dims
    for j in range(dims):
        for k in range(j,dims):
            A[:,col] = x[:,j] * x[:,k] * y_invsigma
            col += 1
    return A

def build_design_vector(y):
    return unumpy.nominal_values(y)/unumpy.std_devs(y)

def parabola_fit(points):
    dims = points[0][0].shape[0]
    
    x = np.array([p[0] for p in points])
    f = np.array([p[1] for p in points])

    A = build_design_matrix(x, f)
    B = build_design_vector(f)[:,np.newaxis] # make column vector

    # Compute best-fit parabola coefficients using a singular value
    # decomposition.
    U, w, V = np.linalg.svd(A, full_matrices=False)
    V = V.T # Flip to convention used by Numerical Recipies
    inv_w = 1.0/w
    inv_w[np.abs(w) < 1e-6] = 0.0
    # Numpy version of Eq 15.4.17 from Numerical Recipies (C edition)
    coeffs = np.zeros(A.shape[1])
    for i in range(len(coeffs)):
        coeffs += (np.dot(U[:,i], B[:,0]) * inv_w[i]) * V[:,i]

    # Chi2 and probability for best fit and quadratic coefficents
    chi2_terms = np.dot(A, coeffs[:,np.newaxis]) - B
    chi2 = (chi2_terms**2).sum()
    ndf = len(points) - (1 + dims + dims * (dims + 1) / 2)
    prob = ROOT.TMath.Prob(chi2, ndf)

    # Covariance is from Eq 15.4.20
    covariance = np.dot(V*inv_w**2, V.T)

    # Pack the coefficients into ufloats
    ufloat_coeffs = uncertainties.correlated_values(coeffs, covariance.tolist())

    # Separate coefficients into a, b, and c
    a = ufloat_coeffs[0]
    b = ufloat_coeffs[1:dims+1]
    c = np.zeros(shape=(dims,dims), dtype=object)
    index = dims + 1
    for i in range(dims):
        for j in range(i, dims):
            c[i,j] = ufloat_coeffs[index]
            c[j,i] = ufloat_coeffs[index]
            if j != i:
                # We combined the redundant off-diagonal parts of c
                # matrix, but now we have to divide by two to 
                # avoid double counting when c is used later
                c[i,j] /= 2.0
                c[j,i] /= 2.0
            index += 1

    return a, np.array(b), c, chi2, prob    
    
def parabola_eval(x, a, b, c):
    if len(x.shape) == 1:
        return a + np.dot(x, b) + np.dot(x, np.dot(c, x.T))
    else:
        y = np.array([a] * x.shape[0])

        for i, xrow in enumerate(x):
            y[i] += np.dot(xrow, b)        
            y[i] += np.dot(xrow, np.dot(c, xrow.T))

        return y
