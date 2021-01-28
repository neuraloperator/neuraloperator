%Radom function from N(m, C) on [0 1] where
%C = sigma^2(-Delta + tau^2 I)^(-gamma)
%with periodic, zero dirichlet, and zero neumann boundary.
%Dirichlet only supports m = 0.
%N is the # of Fourier modes, usually, grid size / 2.
function u = GRF1(N, m, gamma, tau, sigma, type)

if type == "dirichlet"
    m = 0;
end

if type == "periodic"
    my_const = 2*pi;
else
    my_const = pi;
end

my_eigs = sqrt(2)*(abs(sigma).*((my_const.*(1:N)').^2 + tau^2).^(-gamma/2));

if type == "dirichlet"
    alpha = zeros(N,1);
else
    xi_alpha = randn(N,1);
    alpha = my_eigs.*xi_alpha;
end

if type == "neumann"
    beta = zeros(N,1);
else
    xi_beta = randn(N,1);
    beta = my_eigs.*xi_beta;
end

a = alpha/2;
b = -beta/2;

c = [flipud(a) - flipud(b).*1i;m + 0*1i;a + b.*1i];

if type == "periodic"
    uu = chebfun(c, [0 1], 'trig', 'coeffs');
    u = chebfun(@(t) uu(t - 0.5), [0 1], 'trig');
else
    uu = chebfun(c, [-pi pi], 'trig', 'coeffs');
    u = chebfun(@(t) uu(pi*t), [0 1]);
end