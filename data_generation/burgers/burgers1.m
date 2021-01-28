function u = burgers1(init, tspan, s, visc)

S = spinop([0 1], tspan);
S.lin = @(u) + visc*diff(u,2);
S.nonlin = @(u) - 0.5*diff(u.^2);
S.init = init;
u = spin(S,s,1e-4,'plot','off'); 

