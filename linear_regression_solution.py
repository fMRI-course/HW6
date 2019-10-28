""" Understanding least-squares regression
"""
#: Import numerical and plotting libraries
import numpy as np
# Print to four digits of precision
np.set_printoptions(precision=4, suppress=True)
import numpy.linalg as npl
import matplotlib.pyplot as plt



#- Create X design matrix fron column of ones and clammy vector
clammy = [0.389,  0.2  ,  0.241,  0.463,
...           4.585,  1.097,  1.642,  4.972,
...           7.957,  5.585,  5.527,  6.964]

psychopathy = [11.416,   4.514,  12.204,  14.835,
               8.416,   6.563,  17.343, 13.02,
               15.19 ,  11.902,  22.721,  22.324]

n=len(clammy)

X=np.ones((n,2))
X[:,1] = clammy
#- Check whether the columns of X are orthogonal
print('Are the columns orthogonal? ',X[:,0].T.dot(X[:,1]) == '0')
#- Check whether X.T X is invertible
XTX = X.T.dot(X)
XTX_inv = npl.inv(XTX)

#- Calculate (X.T X)^-1 X.T (the pseudoinverse)
pesudo_inv = npl.pinv(X)

#- Calculate least squares fit for beta vector
beta_hat = pesudo_inv.dot(psychopathy)

#- Calculate the fitted values
fitted = X.dot(beta_hat)
#- mean of residuals near zero
residual = psychopathy - fitted
residual_mean = np.mean(residual)
print('Is the residual mean close to zero? ', np.allclose(residual_mean, 0))

#- Residuals orthogonal to design
Residual_orth = X.T.dot(residual)
"""Yes, it is orthogonal."""
#- Copy X to new array X_o
X_o= np.copy(X)

#- Make second column orthononal to first. Confirm orthogonality
# projvw = W.V/||v**2|| * v

X_o[:,1] = X_o[:,1] - np.mean(X_o[:,1])
print('Is the 2nd column orthogonal to the first column? ', np.allclose(X_o[:,0].dot(X_o[:,1]), 0))

""" What is the relationship between the values on the diagonal of
X_o.T.dot(X_o) and the lengths of the vectors in the first and second
columns of X_o?

"""
"""They are equal by chance for the first column since np.sum(X[:,0]) = 12
   The lengh of second column is not equal to the second diagonal values.
   np.sum(X_o[:,1] ** 2) = 2nd diagonal values.

"""

XTX_o = X_o.T.dot(X_o)
square_sum = np.sum(X_o[:,1] ** 2)
np.allclose(XTX_o[0,0], np.sum(X_o[:,0]))
np.allclose(XTX_o[1,1], square_sum)

""" What is the relationship between the values on the diagonal of the
*inverse* of X_o.T.dot(X_o) and the lengths of the vectors in the first
and second columns of X_o?

"""
Inv_XTX = npl.inv(XTX_o)
"""The first element is just the inverse of the length of first vector.
and the second diagonal element is the reverse of np.sum(X[:,1]**2)
"""
#- Make mean-centered version of psychopathy vector
y_c = psychopathy - np.mean(psychopathy)

#- Calculate fit of X_o to y_o
# beta_hat_o = npl.pinv(X_o).dot(y_o)
#- Calculate least squares fit for beta vector
beta_hat_o = npl.pinv(X_o).dot(y_c)

""" Explain the new value of the first element of the parameter estimate
vector.

"""
"""This means the intercept of linear curve is zero."""

fitted  = y_c - X_o.dot(beta_hat_o)
residual_c = y_c - fitted
np.allclose(beta_hat_o[0], np.sum(residual_c ** 2))

#- Correlation coefficient of y_c and the second column of X_o
col_2 = X_o[:,1]
r_xy = np.corrcoef(y_c, col_2)

""" What is the relationship between the correlation coefficient "r_xy"
and the second element in the parameter vector "B_o[1]"?
"""
"""I do not feel there is a relationship between them. r_xy is the correlation
coefficient. But if they are not really good linear fit, we still get a parameter.
"""

#- Fit X_o to psychopathy data
beta_hat_p = npl.pinv(X_o).dot(psychopathy)
""" Explain the relationship between the mean of the psychopathy values
and the first element of the parameter estimate vector.

"""
""""np.mean(psychopathy) = beta_hat_p[0], they are same. """


""" Why is the second value in B_o the same when estimating against "y_c"
and "psychopathy"?
"""
"""The difference between the two model is whether the Y value is mean centered
or not. The center of mean is only related to the first element of beta_hat. So
the second element should be same."""
