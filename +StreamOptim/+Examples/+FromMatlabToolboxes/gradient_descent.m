clear all
close all


%%
n = 100;
phi = 2 * pi * rand(n, 1);
phit = 2 * pi * rand(n, 1) * 1;

x = ones(n, 1) .* exp(1j * phi);
xt = ones(n, 1) .* exp(1j * phit);
q = quality(x, xt);


%% Gradient Descent Example using fminunc in MATLAB
% Define the function
fobj = @(phi) 1 - quality_phi(phi, phit);

% Set options for the optimizer
options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'iter', 'PlotFcns', @optimplotfval);

% Initial guess for x
phi0 = phi;

% Perform the optimization 
[phi, fval, exitflag, output] = fminunc(fobj, phi0, options);

% Display the result
fprintf('Minimum found at x = %.4f, f(x) = %.4f\n', phi, fval);


%%
figure(2), clf, hold on
plot(angle(xt * exp(-1j * angle(xt(1)))), color='k', LineWidth=1.5)
plot(angle(exp(1j * (phi - phi(1)))), marker='.', MarkerSize=20, color='r', LineStyle='none')
grid on
box on