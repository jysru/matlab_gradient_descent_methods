clear all
% close all
clf

import StreamOptim.*


%% Setup problem dimensionality
n = 10;
phi = 2 * pi * rand(n, 1);
phit = 2 * pi * rand(n, 1);

x = ones(n, 1) .* exp(1j * phi);
xt = ones(n, 1) .* exp(1j * phit);
q = StreamOptim.Fitness.quality(x, xt);


%% Gradient Descent using custom toolbox
fobj = @(phi) 1 - StreamOptim.Fitness.quality_phi(phi, phit); % Define the function
alpha = 0.1; % Learning rate
tol = 0; % Norm of variables differences tolerance to stop iterations prematurely
maxIter = 100; % Maximum number of iterations
epsilon = 1e-3; % Perturbation for numerical gradient


%% Setup the finite difference method and the optimizer parameters
opt = StreamOptim.Optims.Optimizer(...
    fobj, phi, alpha, tol=tol, epsilon=epsilon, ...
    maxIter=maxIter, grad_func=@StreamOptim.Gradients.CentralFiniteDifferences, ...
    lb = [], ub = []);


%% Run the optimization algorithm
opt.Run(algorithm='ADMM', plot_each_iter=true);


%% Plot results
opt.history.PlotConvergence("YLim", [0 1], "YScale", 'log')
opt.history.PlotDiffs()
opt.history.PlotGrads()
