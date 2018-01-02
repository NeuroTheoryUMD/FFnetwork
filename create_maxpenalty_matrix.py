from __future__ import division
import numpy as np
import scipy.sparse as sp

def create_maxpenalty_matrix( input_dims, reg_type ):
    """
    Tmat = create_Tikhonov_matrix(stim_params, direction, order)

    Creates a matrix specifying a an L2-regularization operator of the form
    ||T*k||^2. Currently only supports second derivative/Laplacian operations

    INPUTS:
        stim_params: parameter struct associated with the target stimulus.
            must contain .dims field specifying the number of stimulus elements along each dimension
            <.boundary_conds> specifies boundary conditions: Inf is free boundary, 0 is tied to 0, and -1 is periodic
            <.split_pts> specifies an 'internal boundary' over which we dont want to smooth. [direction split_ind split_bnd]
        direction: direction of the derivative relative to the stimulus dimensions. e.g. 1 is along the first dim, 2 is along the second, [1 2] is a laplacian

    OUTPUTS:
        Tmat: sparse matrix specifying the desired Tikhonov operation

    # Boundary conditions would be ideally a dictionary with each reg type listed. Assumed infinity without
    #   Currently not implemented

    The method of computing sparse differencing matrices used here is adapted from
    Bryan C. Smith's and Andrew V. Knyazev's function "laplacian", available
    here: http://www.mathworks.com/matlabcentral/fileexchange/27279-laplacian-in-1d-2d-or-3d
    Written in Matlab by James McFarland, adapted into python by Dan Butts
    """

    allowed_reg_types = ['max', 'max_filt', 'max_space', 'centralizer1']
    #assert (ischar(reg_type) && ismember(reg_type, allowed_reg_types), 'not an allowed regularization type');

    num_filt = input_dims[0] # first dimension is assumed to represent filters
    num_pix = input_dims[1]*input_dims[2] # additional dimensions are spatial (Nx and Ny)
    NK = num_filt * num_pix

    rmat = np.zeros([NK,NK], dtype=np.float32)
    
    if reg_type is 'max':
        # Simply subtract the diagonal from all-ones
        rmat = np.ones([NK,NK],dtype=np.float32) - np.eye(NK, dtype=np.float32)

    elif reg_type is 'max_filt':
        ek = np.ones([num_filt, num_filt], dtype=np.float32) - np.eye(num_filt, dtype=np.float32)
        rmat = np.kron(np.eye(num_pix), ek)

    elif reg_type is 'max_space':
        ex = np.ones([num_pix, num_pix]) - np.eye(num_pix)
        rmat = np.kron(ex, np.eye(num_filt, dtype=np.float32))
        
    elif reg_type is 'centralizer1':
        for i in range(NK):
            rmat[i, i] = np.power(i//input_dims[0] - num_pix/2 + 0.5, 2, dtype=float)
    else:
        print('Havent made this type of reg yet. What you are getting wont work.')

    return rmat
