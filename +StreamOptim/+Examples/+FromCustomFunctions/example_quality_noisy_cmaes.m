clear all
% close all
clf

import StreamOptim.*

%% Setup problem dimensionality
n = 20;
cost_function_noise_std = 0.005 * 1;
phit = 2 * pi * rand(1, n);
phi = 2 * pi * rand(1, n);

%% Gradient Descent using custom toolbox
fobj = @(phi) 1 - (StreamOptim.Fitness.quality_phi(phi, phit) - abs(cost_function_noise_std * randn()) ); % Define the function





%% Setup the finite difference method and the optimizer parameters
m = 100;
initial_points = ones(n, m) .* exp(1j * 2 * pi * rand(n, m));
initial_points = angle(initial_points).';



%%
opts = StreamOptim.Optims.cmaes;
opts.StopFitness = 1e-3;
opts.LBounds = -pi * ones(n, 1);
opts.UBounds = +pi * ones(n, 1);
opts.LogPlot = 'on';
opts.MaxIter = 20;
opts.PopSize = 10;
% opts.StopFunEvals = 200;

%%

opt = StreamOptim.Optims.cmaes( ...
    'StreamOptim.Fitness.cost_cophasing', phi, 2, opts);
    