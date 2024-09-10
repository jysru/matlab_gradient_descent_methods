clear all
% close all

clf


import StreamOptim.*

%%
random_tries = 10;

n = 10;
phi = 2 * pi * rand(n, 1);
phit = 2 * pi * rand(n, 1);

x = ones(n, 1) .* exp(1j * phi);
xt = ones(n, 1) .* exp(1j * phit);
q = StreamOptim.Fitness.quality(x, xt);


%% Gradient Descent Example using fminunc in MATLAB
% Define the function
fobj = @(phi) 1 - StreamOptim.Fitness.quality_phi(phi, phit);

% Learning rate
alpha = 0.1;

% Tolerance
tol = 0;

% Maximum number of iterations
maxIter = 100;

% Perturbation for numerical gradient
epsilon = 1e-3;

opt = StreamOptim.Optims.Optimizer(...
    fobj, phi, alpha, tol=tol, epsilon=epsilon, ...
    maxIter=maxIter, grad_func=@StreamOptim.Gradients.CentralFiniteDifferences, ...
    lb = [], ub = []);

opt.Run(algorithm='ADMM', plot_each_iter=true);

%%
opt.history.PlotConvergence("YLim", [0 1], "YScale", 'log')
opt.history.PlotDiffs()
opt.history.PlotGrads()
