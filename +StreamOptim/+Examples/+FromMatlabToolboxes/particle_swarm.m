clear all
close all


%%
n = 10;
phi = 2 * pi * rand(n, 1);
phit = 2 * pi * rand(n, 1) * 1;

x = ones(n, 1) .* exp(1j * phi);
xt = ones(n, 1) .* exp(1j * phit);
q = quality(x, xt);


%% Particle Swarm MATLAB
% Define the function
fobj = @(phi) 1 - quality_phi(phi, phit);

lb = -5*pi * ones(n, 1);
ub = +5*pi * ones(n ,1);

options = optimoptions('particleswarm', 'SwarmSize', 1000, 'Display', 'iter', MaxStallIterations=50);

[phi, fval, exitflag] = particleswarm(fobj, n, lb, ub, options);

% % Display the result
fprintf('Minimum found at x = %.4f, f(x) = %.4f\n', phi, fval);



%%
figure(2), clf, hold on
plot(angle(xt * exp(-1j * angle(xt(1)))), color='k', LineWidth=1.5)
plot(angle(exp(1j * (phi - phi(1)))), marker='.', MarkerSize=20, color='r', LineStyle='none')
grid on
box on
