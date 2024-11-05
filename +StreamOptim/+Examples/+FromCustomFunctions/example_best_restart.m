clear all
% close all
clf

import StreamOptim.*


%% Setup problem dimensionality
n = 10;
variables_noise_std = 0.0;
cost_function_noise_std = 0.05 * 1;
phit = 2 * pi * rand(n, 1);
phi = 2 * pi * rand(n, 1);
% phi = phit + 1 * randn(n, 1);

x = ones(n, 1) .* exp(1j * phi);
xt = ones(n, 1) .* exp(1j * phit);
q = StreamOptim.Fitness.quality(x, xt);


%% Gradient Descent using custom toolbox
fobj = @(phi) abs(1 - (StreamOptim.Fitness.quality_phi(phi, phit) + cost_function_noise_std * randn())); % Define the function
alpha = 1e-1; % Learning rate
tol = 0; % Norm of variables differences tolerance to stop iterations prematurely
maxIter = 100; % Maximum number of iterations
epsilon = 3e-1; % Perturbation for numerical gradient
algorithm = 'Adam';
% algorithm = 'Adam';


%% Setup the finite difference method and the optimizer parameters
opt = StreamOptim.Optims.Optimizer(...
    fobj, phi, alpha, tol=tol, epsilon=epsilon, ...
    maxIter=maxIter, grad_func=@StreamOptim.Gradients.ParallelRandomSignCentralFiniteDifferences, ...
    lb = [], ub = []);


%% Evaluate the noise
opt.EvaluateNoise(perturb=epsilon, max_iter=200)
opt.history.PlotNoiseWithPerturbs();


%% Run the optimization algorithms
opt.Run(algorithm=algorithm, plot_each_iter=false, add_variables_noise_each_iter=false, noise_std=variables_noise_std);

%% Plot results
opt.history.PlotSteps()
opt.history.PlotConvergence()
opt.history.PlotConvergence()


%%
best = opt.history.GetBest();
disp(best.fvals)
disp(fobj(best.x))

opt2 = StreamOptim.Optims.Optimizer(...
    fobj, best.x, best.alpha/10, tol=tol, epsilon=epsilon, ...
    maxIter=maxIter, grad_func=@StreamOptim.Gradients.ParallelRandomSignCentralFiniteDifferences, ...
    lb = [], ub = []);

%% Evaluate the noise
opt2.EvaluateNoise(perturb=epsilon, max_iter=200)
opt2.history.PlotNoiseWithPerturbs("FigureNumber",11);

%% Run the optimization algorithms
opt2.Run(algorithm=algorithm, plot_each_iter=false, add_variables_noise_each_iter=false, noise_std=variables_noise_std);


%% Plot results
opt2.history.PlotSteps(FigureNumber=12)
opt2.history.PlotConvergence(FigureNumber=13)
opt2.history.PlotConvergence(FigureNumber=13)

