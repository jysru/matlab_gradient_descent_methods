clear all
% close all
clf

import StreamOptim.*

%% Setup problem dimensionality
n = 10;
cost_function_noise_std = 0.005 * 1;
phit = 2 * pi * rand(1, n);
phi = 2 * pi * rand(1, n);

%% Gradient Descent using custom toolbox
fobj = @(phi) 1 - (StreamOptim.Fitness.quality_phi(phi, phit) - abs(cost_function_noise_std * randn()) ); % Define the function

%% Setup the finite difference method and the optimizer parameters
m = 100;
initial_points = ones(n, m) .* exp(1j * 2 * pi * rand(n, m));
initial_points = angle(initial_points).';

% options = optimoptions('particleswarm','SwarmSize',100,'HybridFcn',@fmincon);
options = optimoptions( ...
    'simulannealbnd', 'MaxStallIterations',20, 'PlotFcn','saplotbestf', 'FunctionTolerance', 0.001);
[x,fval,exitFlag,output] = simulannealbnd(fobj, phi, -pi*ones(n,1), +pi*ones(n,1), options);