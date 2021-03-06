import numpy as np
import scipy.sparse as sp

def create_Tikhonov_matrix( stim_dims, reg_type, boundary_conditions=None ):
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


    nLags = stim_dims[0] # first dimension is assumed to represent time lags
    nPix = stim_dims[1]*stim_dims[2] # additional dimensions are spatial (Nx and Ny)
    allowed_reg_types = ['d2xt', 'd2x', 'd2t']

    #assert (ischar(reg_type) && ismember(reg_type, allowed_reg_types), 'not an allowed regularization type');

    #has_split = ~isempty(stim_params.split_pts);
    et = np.ones( [1,nLags], dtype=np.float32 )
    ex = np.ones( [1,stim_dims[1]], dtype=np.float32 )
    ey = np.ones( [1,stim_dims[2]], dtype=np.float32 )

    # Boundary conditions (currently not implemented)
    #if isinf(stim_params.boundary_conds(1)) # if temporal dim has free boundary
    #et[0, [0, -1]] = 0  # constrain temporal boundary to zero: all else are free
    #if isinf(stim_params.boundary_conds(2)) # if first spatial dim has free boundary
    ex[0, [0, -1]] = 0
    #if isinf(stim_params.boundary_conds(3)); # if second spatial dim has free boundary
    ey[0, [0, -1]] = 0

    if nPix == 1:       # for 0-spatial-dimensional stimuli can only do temporal

        assert reg_type is 'd2t', 'Can only do temporal reg for stimuli without spatial dims'

        Tmat = sp.spdiags( np.concatenate( (et,-2*et,et),axis=0), [-1,0,1], nLags, nLags )
        #if stim_params.boundary_conds(1) == -1 # if periodic boundary cond
        #    Tmat(end, 1) = 1;
        #    Tmat(1, end) = 1;

    elif stim_dims[2] == 1: # for 1 - spatial dimensional stimuli
        if reg_type is 'd2t':
            assert nLags>1, 'No d2t regularization possible with no lags.'

            #D1t = spdiags([et - 2 * et et], [-1 0 1], nLags, nLags)';
            D1t = sp.coo_matrix.transpose( sp.coo_matrix(
                sp.spdiags( np.concatenate((et,-2*et,et), axis=0), [-1,0,1], nLags, nLags) ) )
            #    if stim_params.boundary_conds(1) == -1 % if periodic boundary cond
            #        D1t(end, 1) = 1;
            #        D1t(1, end) = 1;
            Ix = sp.eye(stim_dims[1])
            Tmat = sp.kron(Ix, D1t)

        elif reg_type is 'd2x':
            It = sp.eye(nLags)
            # Matlab code: D1x = spdiags([ex -2*ex ex], [-1 0 1], nPix(1), nPix(1))';
            D1x = sp.coo_matrix.transpose( sp.coo_matrix(
                sp.spdiags( np.concatenate( (ex,-2*ex,ex), axis=0), [-1,0,1], stim_dims[1], stim_dims[1] ) ))
            #if stim_params.boundary_conds(2) == -1 % if periodic boundary cond
            #    D1x(end, 1) = 1;
            #    D1x(1, end) = 1;

            Tmat = sp.kron( D1x, It )

        elif reg_type is 'd2xt':
            # D1t = spdiags([et - 2 * et et], [-1 0 1], nLags, nLags)';
            D1t = sp.coo_matrix.transpose(sp.coo_matrix(
                sp.spdiags( np.concatenate( (et,-2*et,et), axis=0), [-1,0,1], nLags,nLags ) ))
            #if stim_params.boundary_conds(1) == -1 % if periodic boundary cond
            #    D1t(end, 1) = 1;
            #    D1t(1, end) = 1;

            #D1x = spdiags([ex - 2 * ex ex], [-1 0 1], nPix(1), nPix(1))';
            D1x = sp.coo_matrix.transpose( sp.coo_matrix(
                sp.spdiags( np.concatenate( (ex,-2*ex,ex), axis=0), [-1,0,1], stim_dims[1], stim_dims[1] ) ))
            #if stim_params.boundary_conds(2) == -1 % if periodic boundary cond
            #    D1x(end, 1) = 1;
            #    D1x(1, end) = 1;

            It = sp.eye(nLags)
            Ix = sp.eye(stim_dims[1])
            Tmat = sp.kron(Ix, D1t) + sp.kron(D1x, It)

    else: # for stimuli with 2-spatial dimensions
        if reg_type is 'd2t':
            assert nLags>1, 'No d2t regularization possible with no lags.'
            # D1t = spdiags([et - 2 * et et], [-1 0 1], nLags, nLags)';
            D1t = sp.coo_matrix.transpose( sp.coo_matrix(
                sp.spdiags( np.concatenate((et,-2*et,et), axis=0), [-1,0,1], nLags, nLags) ) )
            #if stim_params.boundary_conds(1) == -1 % if periodic boundary cond
            #    D1t(end, 1) = 1;
            #    D1t(1, end) = 1;

            Ix = sp.eye(stim_dims[1])
            Iy = sp.eye(stim_dims[2])
            Tmat = sp.kron( Iy, sp.kron(Ix,D1t) )

        elif reg_type is 'd2x':
            It = sp.eye(nLags)
            Ix = sp.eye(stim_dims[1])
            Iy = sp.eye(stim_dims[2])
            # D1x = spdiags([ex - 2 * ex ex], [-1 0 1], nPix(1), nPix(1))';
            D1x = sp.coo_matrix.transpose( sp.coo_matrix(
                sp.spdiags( np.concatenate( (ex,-2*ex,ex), axis=0), [-1,0,1], stim_dims[1], stim_dims[1] ) ))
            #if stim_params.boundary_conds(2) == -1 % if periodic boundary cond
            #    D1x(end, 1) = 1
            #    D1x(1, end) = 1

            # D1y = spdiags([ey - 2 * ey ey], [-1 0 1], nPix(2), nPix(2))';
            D1y = sp.coo_matrix.transpose( sp.coo_matrix(
                sp.spdiags( np.concatenate( (ey,-2*ey,ey), axis=0), [-1,0,1], stim_dims[2], stim_dims[2] ) ))
            #if stim_params.boundary_conds(3) == -1 % if periodic boundary cond
            #    D1y(end, 1) = 1;
            #    D1y(1, end) = 1;

            Tmat = sp.kron( Iy, sp.kron(D1x,It) ) + sp.kron( D1y, sp.kron(Ix,It) )

        elif reg_type is 'd2xt':
            It = sp.eye(nLags)
            Ix = sp.eye(stim_dims[1])
            Iy = sp.eye(stim_dims[2])
            # D1t = spdiags([et - 2 * et et], [-1 0 1], nLags, nLags)';
            D1t = sp.coo_matrix.transpose( sp.coo_matrix(
                sp.spdiags( np.concatenate((et,-2*et,et), axis=0), [-1,0,1], nLags, nLags) ) )
            #if stim_params.boundary_conds(1) == -1 # if periodic boundary cond
            #    D1t(end, 1) = 1
            #    D1t(1, end) = 1

            # D1x = spdiags([ex - 2 * ex ex], [-1 0 1], nPix(1), nPix(1))';
            D1x = sp.coo_matrix.transpose( sp.coo_matrix(
                sp.spdiags( np.concatenate( (ex,-2*ex,ex), axis=0), [-1,0,1], stim_dims[1], stim_dims[1] ) ))
            #if stim_params.boundary_conds(2) == -1 % if periodic boundary cond
            #    D1x(end, 1) = 1;
            #    D1x(1, end) = 1;

            # D1y = spdiags([ey - 2 * ey ey], [-1 0 1], nPix(2), nPix(2))';
            D1y = sp.coo_matrix.transpose( sp.coo_matrix(
                sp.spdiags( np.concatenate( (ey,-2*ey,ey), axis=0), [-1,0,1], stim_dims[2], stim_dims[2] ) ))
            #if stim_params.boundary_conds(3) == -1 % if periodic boundary cond
            #    D1y(end, 1) = 1;
            #    D1y(1, end) = 1;

            Tmat = sp.kron( D1y, sp.kron(Ix,It) ) + sp.kron( Iy, sp.kron(D1x,It) ) + sp.kron( Iy, sp.kron(Ix,D1t) )

        else:
            print('Unsupported reg type.')
            Tmat = None

    Tmat = Tmat.toarray()  # make dense matrix before sending home

    return Tmat
