clear all
close all


%%
n = 100;
iters = 10;
phi = 2 * pi * rand(n, 1);
phit = 2 * pi * rand(n, 1) * 1;

x = ones(n, 1) .* exp(1j * phi);
xt = ones(n, 1) .* exp(1j * phit);
q = quality(x, xt);


%% Gradient Descent Example using fminunc in MATLAB
% Define the function
fobj = @(phi) 1 - quality_phi(phi, phit);

% Set options for the optimizer
options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', 'off', 'MaxIter', 0);
fvals =  nan(iters, 1);

for i=1:iters
    % Perform the optimization 
    [phi, fval, exitflag, output] = fminunc(fobj, phi, options);
    fvals(i) = fval;
end

% Plot results
plot(fvals, 'Marker', '.', 'MarkerSize', 15, 'LineStyle', 'none')
title('Optimization dynamics')
xlabel('Iteration #')
ylabel('1 - Q')
grid on
ylim([0 1])
yscale('log')



