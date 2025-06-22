function [ArQr, AlQl] = FCGram_Matrices(d, C, Z, E, n_over, modes_to_reduce, num_digits)
%FCGram_Matrices Compute Gram matrices for Fourier Continuation with FC-Gram
%
% This function computes the Gram matrices required for Fourier Continuation 
% with FC-Gram algorithm. The matrices enable efficient extension of 
% non-periodic functions to periodic ones for spectral differentiation.
%
% Parameters
% ----------
% d : int
%     Number of matching points (degree of approximation)
% C : int
%     Number of continuation points (must be even)
% Z : int
%     Number of zero padding points for smooth extension
% E : int
%     Number of extra points for numerical stability
% n_over : int
%     Oversampling factor for fine grid construction
% modes_to_reduce : int
%     Number of modes to reduce in SVD truncation
% num_digits : int
%     Number of digits for symbolic precision in computations
%
% Returns
% -------
% ArQr : double matrix
%     Right boundary continuation matrix (C x d)
% AlQl : double matrix
%     Left boundary continuation matrix (C x d)
%
% Algorithm
% ---------
% 1. Construct monomial basis on coarse grid
% 2. Orthonormalize using QR decomposition with full re-orthogonalization
% 3. Evaluate basis on fine grid with oversampling
% 4. Build trigonometric basis for FC approximation
% 5. Compute SVD and solve for coefficients
% 6. Evaluate FC matrix at continuation points
% 7. Construct boundary continuation matrices with flip operations
%
% Notes
% -----
% The function automatically saves the computed matrices to a .mat file
% in the format: FCGram_data_d{d}_C{C}.mat
%
% References
% ----------
% [1] Amlani, F., & Bruno, O. P. (2016). An FC-based spectral solver for
%     elastodynamic problems in general three-dimensional domains. 
%     Journal of Computational Physics, 307, 333-354.
%
% Authors: Daniel Leibovici, Valentin Duruisseaux
% Email: dleibovi@caltech.edu, vduruiss@caltech.edu
    
    % Set tolerance for numerical operations
    tol = 1e-16;
    
    % Set symbolic precision
    digits(num_digits);
    
    % Total number of coarse points in the domain
    N_coarse = sym(d + C + Z + E);
    
    % Order of SVD truncation (number of modes to keep)
    svd_order = sym(N_coarse - 2 * modes_to_reduce);
    
    % Check if svd_order is valid
    if svd_order <= 0
        error('Invalid parameters: svd_order = %d <= 0. Please reduce modes_to_reduce or increase N_coarse (d+C+Z+E).\nCurrent values: d=%d, C=%d, Z=%d, E=%d, N_coarse=%d, modes_to_reduce=%d', ...
              double(svd_order), d, C, Z, E, double(N_coarse), modes_to_reduce);
    end
    
    % Create flip matrices for boundary condition handling
    % F1: flip matrix for left boundary (d x d)
    % F2: flip matrix for right boundary (C x C)
    F1 = flipud(sym(eye(d)));
    F2 = flipud(sym(eye(C)));
    
    % Reference grid parameters
    N0 = sym(101);                    % Number of reference grid points
    h0 = sym(1 / (N0 - 1));           % Reference grid spacing
    x0 = sym(1 - (d - 1)*h0);         % Left boundary of interpolation domain
    midpoint = x0 + (d + C)*h0;       % Midpoint for zero padding region
    
    % Coarse interpolation points: where we have function values
    interp_coarse = x0 + h0 * (0 : d - 1).';
    
    % Fine grid points: oversampled version of interpolation domain
    interp_fine = linspace(x0, 1, n_over*(d - 1) + 1).';
    
    % Zero padding points: where function is extended to zero
    zero_fine = midpoint + linspace(0, (Z - 1)*h0, n_over*(Z - 1) + 1).';
    
    % Continuation points: where FC approximation is evaluated
    continuation = x0 + h0 * (d : d + C - 1).';
    
    % Build monomial basis matrix P_coarse (d x d)
    % Each column contains powers of x: [1, x, x^2, ..., x^(d-1)]
    P_coarse = sym(zeros(d, d));
    for i = 0 : d - 1
        P_coarse(:, i + 1) = interp_coarse.^i;
    end
    
    % Compute QR decomposition to get orthonormal basis Q (and R is upper triangular)
    [Q, R] = mgs(P_coarse);
    
    % Build monomial basis on fine grid
    P_fine = sym(zeros(n_over*(d - 1) + 1, d));
    for i = 0 : d - 1
       P_fine(:, i + 1) = interp_fine.^i;
    end
    
    % Evaluate orthonormal basis Q on fine grid using R from QR decomposition
    Q_fine = P_fine / R;
    
    % Combine fine grid and zero padding points for FC approximation
    X = [interp_fine; zero_fine];
    
    % Determine wavenumbers for cosine and sine basis functions
    if mod(svd_order, 2) == 0
        % Even number of modes: equal number of cosines and sines
        k_cos = 0 : svd_order/2 - 1;      % Cosine wavenumbers 
        k_sin = 1 : svd_order/2 - 1;      % Sine wavenumbers 
    else
        % Odd number of modes: one extra cosine mode
        k_cos = 0 : (svd_order - 1)/2;   
        k_sin = 1 : (svd_order - 1)/2;   
    end
    
    % Check if wavenumber arrays are valid
    if isempty(k_cos) || isempty(k_sin)
        error('Invalid svd_order: wavenumber arrays are empty. svd_order = %d', double(svd_order));
    end
    
    % Build matrix of trigonometric basis functions
    % Columns: [cos(0), cos(1), ..., cos(k_cos_max), sin(1), ..., sin(k_sin_max)]
    C_mat = [cos(2*pi*X*k_cos / ((N_coarse - 1)*h0)), ...
             sin(2*pi*X*k_sin / ((N_coarse - 1)*h0))];
    
    % Compute SVD of trigonometric basis matrix
    [U, S, V] = svd(C_mat, 'econ');
    
    % Initialize coefficient matrix
    Coeffs = sym(zeros(size(C_mat, 2), d));
    
    % Get singular values and their inverses
    delta = diag(S);
    delta_inv = 1 ./ delta;
    
    % Solve for coefficients of each basis function
    for i = 1 : d
        % Right-hand side: orthonormal basis function extended with zeros
        b = [Q_fine(:, i); zeros(n_over*(Z - 1) + 1, 1)];
        
        % Solve using SVD: Coeffs(:,i) = V * inv(S) * U' * b
        Coeffs(:, i) = V * (delta_inv .* (U' * b));
    end
    
    % Evaluate FC approximation matrix at continuation points
    % A = trigonometric_basis(continuation) * coefficients
    A = vpa([cos(2*pi*continuation * k_cos / ((N_coarse - 1) * h0)), ...
             sin(2*pi*continuation * k_sin / ((N_coarse - 1) * h0))] * Coeffs);
    
    % Construct matrices for handling boundary conditions
    % ArQr: Right boundary continuation (standard)
    % AlQl: Left boundary continuation (with flip matrices for symmetry)
    ArQr = A * Q.';
    AlQl = F2 * A * Q.' * F1;

    % Convert matrices to double precision
    ArQr = double(ArQr);
    AlQl = double(AlQl);

    % Save all matrices to a single .mat file
    save(sprintf('./FCGram_matrices/FCGram_data_d%d_C%d.mat', d, C), 'ArQr', 'AlQl');

    % Print success message
    fprintf('Matrices ArQr, AlQl saved to ./FCGram_matrices/FCGram_data_d%d_C%d.mat !\n', d, C);
    
end



function [Q, R] = mgs(A)
%mgs Modified Gram-Schmidt routine with full re-orthogonalization
%
% This local function implements the Modified Gram-Schmidt algorithm
% with full re-orthogonalization to ensure numerical stability and
% orthogonality of the resulting Q matrix.
%
% Parameters
% ----------
% A : symbolic matrix
%     Input matrix of size m x n to decompose
%   
% Returns
% -------
% Q : symbolic matrix
%     Orthonormal matrix of size m x n
% R : symbolic matrix
%     Upper triangular matrix of size n x n
%
% Algorithm
% ---------
% 1. Initialize Q and R matrices
% 2. For each column j of A:
%    a. Project onto previous orthonormal vectors
%    b. Perform full re-orthogonalization for numerical stability
%    c. Normalize to get unit vector
% 3. Store projection coefficients in R matrix
%
% Notes
% -----
% The full re-orthogonalization step safeguards against roundoff errors
% and ensures high-quality orthogonality of the resulting Q matrix.

    m = size(A,1);
    n = size(A,2);
    Q = sym(zeros(m,n));
    R = sym(zeros(n,n));
    
    % Initialize first column
    R(1,1) = sqrt(A(:,1).' * A(:,1));
    Q(:,1) = A(:,1)/R(1,1);
    
    % Process remaining columns
    for j=2:n
        Q(:, j) = A(:,j);
        
        % First orthogonalization pass
        for i=1:j-1
            R(i,j) = Q(:, j).'*Q(:,i);
            Q(:, j) = Q(:, j) - R(i,j)*Q(:,i);
        end
        
        % Full re-orthogonalization to safeguard against roundoff errors and ensure orthogonality
        for i = 1 : j-1
            proj_subspace = Q(:,i).' * Q(:,j); 
            R(i, j) = R(i, j) + proj_subspace;        
            Q(:,j) = Q(:,j) - proj_subspace * Q(:,i);
        end 
        
        % Normalize to get unit vector
        R(j, j) = sqrt(Q(:, j).' * Q(:, j));
        Q(:,j) = Q(:, j)/R(j,j);
    end
end