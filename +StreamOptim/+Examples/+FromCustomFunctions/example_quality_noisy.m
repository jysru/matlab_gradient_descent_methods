clear all
% close all
clf

import StreamOptim.*


%% Setup problem dimensionality
n = 10;
variables_noise_std = 0.0;
cost_function_noise_std = 0.01 * 1;
phit = 2 * pi * rand(n, 1);
phi = 2 * pi * rand(n, 1);
phi = phit + 2 * rand(n, 1);

x = ones(n, 1) .* exp(1j * phi);
xt = ones(n, 1) .* exp(1j * phit);
q = StreamOptim.Fitness.quality(x, xt);


%% Gradient Descent using custom toolbox
fobj = @(phi) 1 - (StreamOptim.Fitness.quality_phi(phi, phit) + cost_function_noise_std * randn()); % Define the function
alpha = 0.1; % Learning rate
tol = 0; % Norm of variables differences tolerance to stop iterations prematurely
maxIter = 100; % Maximum number of iterations
epsilon = 1e-1; % Perturbation for numerical gradient
algorithm = 'ADMM';


%% Setup the finite difference method and the optimizer parameters
opt = StreamOptim.Optims.Optimizer(...
    fobj, phi, alpha, tol=tol, epsilon=epsilon, ...
    maxIter=maxIter, grad_func=@StreamOptim.Gradients.ParallelRandomSignCentralFiniteDifferences, ...
    lb = [], ub = []);


%% Evaluate the noise
opt.EvaluateNoise(perturb=0.1, max_iter=1000)
opt.history.PlotNoiseAndSteps();


%% Run the optimization algorithms
opt.Run(algorithm=algorithm, plot_each_iter=false, add_variables_noise_each_iter=true, noise_std=variables_noise_std);

%% Plot results
figure(1); clf, hold on
plot(opt.history.fvals, 'Marker', '.', 'MarkerSize', 10)
title([opt.history.algorithm ' optimization: Convergence'])
xlabel('Iteration #')
ylabel('Cost function')
grid on, box on
ylim([0, 1])
% set(gca, 'YScale', 'log')


opt.history.PlotDiffs()
opt.history.PlotGrads()


