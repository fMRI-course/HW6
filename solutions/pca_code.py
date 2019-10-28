""" PCA exercise
    10/27/19.
"""

#: import common modules
import numpy as np  # the Python array package
import matplotlib.pyplot as plt  # the Python plotting package
# Display array values to 6 digits of precision
np.set_printoptions(precision=4, suppress=True)

#: import numpy.linalg with a shorter name
import numpy.linalg as npl
import nibabel as nib
#- Load the image 'ds114_sub009_t2r1.nii' with nibabel
#- Get the data array from the image
img=nib.load('ds114_sub009_t2r1.nii', mmap=False)
data=img.get_data()
#- Make variables:
#- 'vol_shape' for shape of volumes
#- 'n_vols' for number of volumes
vol_shape=data.shape[:-1]
n_vols=data.shape[-1]
print('The shape of the data ds114_sub009_t2r1.nii is: ', vol_shape)
print('The number of volumes is ',n_vols)
#- Slice the image data array to give array with only first two
#- volumes
vol0=data[...,0]
vol1=[...,1]
vol_first_two=data[...,0:2]
print("The shape of first two volumes is", vol_first_two.shape)
#- Set N to be the number of voxels in a volume
N = np.prod(vol_shape)
#N=len(vol0.ravel())
print("The number of pixels in a volume is ", N)
#- Reshape to 2D array with first dimension length N
data_2d=vol_first_two.reshape((N,2))
print("The shape of reshaped data of first two volumes is ", data_2d.shape)
#- Transpose to 2 by N array
data_2d_T=data_2d.T#
#- Calculate the mean across columns
data_2d_mean=data_2d_T.mean(axis=1)

#- Row means copied N times to become a 2 by N array
row_means = np.outer(data_all_mean，np.ones(N))

#data_2d_row_mean=data_2d_T.mean(axis=1)
#data_2d_cMean=data_2vol_mean.reshape((2,N))
print('The shape of row means copied N time is ',data_2d_cMean.shape)
# print('The type of data.type is ', type(data.shape))
#- Subtract the means for each row, put the result into X
#- Show the means over the columns, after the subtraction
X = data_2d_T - row_means
means=X.mean(axis=0)
print('The shape means variable is ', means.shape)
plt.figure(1)
plt.hist(means)

#- Plot the signal in the first row against the signal in the second
plt.figure(2)
plt.plot(X[0],X[1], '+')
# plt.show()
#- Calculate unscaled covariance matrix for X
unscaled_var=X.dot(X.T)

#- Use SVD to return U, S, VT matrices from unscaled covariance
U,S,VT=npl.svd(unscaled_var, full_matrices=False)
#- Show that the columns in U each have vector length 1
np.sum(U ** 2, axis = 0)

uni=[]
for i in range(len(U)):
    uni.append(U[i,0]**2 + U[i,1]**2)
    test=np.allclose(uni[i], 1.0)
    print('Is the length of %ith column vector equal to 1' %i, test)
# print(uni)

#U_dot=U[:,0].dot(U[:,1])
#orth_test=np.allclose(U_dot, 0.0)
#Test the orthogonality of two columns of U.
# print(U_dot)
np.allclose(U[:, 0].dot(U[:, 1]), 0)
print("Is The two columns of U othogonal? ", orth_test)
#- Confirm tranpose of U is inverse of U
#orth_rev_test=np.allclose(U.T, npl.inv(U))
np.allclose(np.eye(2), U.T.dot(U))
print("Is the tranpose of U equal to inverse of U?", orth_test)
#- Show the total sum of squares in X
#- Is this (nearly) the same as the sum of the values in S?
Sum_X=np.sum(X**2)
print("Total sum of squares in X is", Sum_X)
Sum_test=np.allclose(Sum_X, np.sum(S))
print("Is the sum of squares in X is equal to the sum of squeares of X from svd?", Sum_test)
#- Plot the signal in the first row against the signal in the second
#- Plot line corresponding to a scaled version of the first principal component
#- (Scaling may need to be negative)
plt.figure(3)
plt.plot(X[0,:], X[1,:],'*')
plt.plot([0,-4000*U[0,0]],[0, -4000*U[1,0]], linewidth=5.0)
# plt.show

#- Calculate the scalar projections for projecting X onto the
#- vectors in U.
#- Put the result into a new array C.
C=U.dot(X)
print('The shape of C is ', C.shape)
#- Transpose C
#- Reshape the first dimension of C to have the 3D shape of the
#- original data volumes.
C_T=C.T
C_vol=C_T.reshape((64, 64, 30,2))
#C_vol = C_T.reshape(vol_shape+(2,))
plt.figure(4)
plt.imshow(C_vol[:, :, 15,0])
# plt.show()
#- Break 4D array into two 3D volumes
print("The ndim of my 4D data", data.shape)
vol_part_first=data[:,:,:,0:15]
vol_part_two=data[:,:,:,16:31]
#- Show middle slice (over third dimension) from scalar projections
#- for first component
new_vol1=vol_part_first[...,8].ravel()
new_vol2=vol_part_two[...,8].ravel()
new_vol1_vol2=np.vstack((new_vol1, new_vol2))
C_part_first=U[0,:].reshape(1,2).dot(new_vol1_vol2)
Projected_C=U.T[0,:].reshape(2,1).dot(C_part_first)
print("Projected C values for middle slice in first compment has ", C_part_first.shape)
print('Projected data is ', Projected_C.shape)
# print("Peiwu")
#- Show middle slice (over third dimension) from scalar projections
#- for second component

#- Reshape first dimension of whole image data array to N, and take
#- transpose
data_all_2d=data.reshape(N, n_vols)
print("The shape of data_all_2d is ", data_all_2d.shape)
data_all_T=data_all_2d.T
#- Calculate mean across columns
#- Expand to (173, N) shape using np.outer
#- Subtract from data array to remove mean over columns (row means)
#- Put result into array X
data_all_mean=data_all_T.mean(axis=1)
one_n=np.ones(N)
mean_expand=np.outer(data_all_mean, one_n)
X_all=data_all_T-mean_expand
#- Calculate unscaled covariance matrix of X

unscaled_var_all=X_all.dot(X_all.T) #We did this at the begining
all_U, all_s, all_VT = npl.svd(unscaled_var_all)
#- Use subplots to make axes to plot first 10 principal component
#- vectors
#- Plot one component vector per sub-plot.

fig, axes = plt.subplots(10,1)
fig = plt.figure(5)
for i in range(10):
    axes[i].plot(all_U[i,:])
# plt.show()

#- Calculate scalar projections for projecting X onto U
#- Put results into array C.
C_all=all_U.dot(data_all_T)
#- Transpose C
#- Reshape the first dimension of C to have the 3D shape of the
#- original data volumes.
C_all_T=C_all.T
C_all_3D=C_all_T.reshape(64,64, 30, n_vols)
#- Show middle slice (over third dimension) of first principal
#- component volume
plt.figure(6)
plt.imshow(C_all_3D[:,:,15,0])
# plt.show()
#- Make the mean volume (mean over the last axis)
#- Show the middle plane (slicing over the third axis)
mean_C=C_all_3D.mean(axis=3)
plt.figure(7)
plt.imshow(mean_C[:,:,15])
# plt.show()
#- Show middle plane (slice over third dimension) of second principal
#- component volume
plt.figure(8)
plt.imshow(C_all_3D[:,:,15,1])

#- Show middle plane (slice over third dimension) of third principal
#- component volume
plt.figure(9)
plt.imshow(C_all_3D[:,:,15,2])
plt.show()
