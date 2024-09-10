function [grad, fevals, f_perturb, f_0] = ParallelRandomSignFiniteDifferences(func, x, epsilon)
    % Compute the numerical gradient from central finite differences for all components in parallel with random signs
    %
    % Inputs:
    % func      - Handle to the objective function to minimize.
    % x0        - Variables vector.
    % alpha     - Learning rate (step size).
    % tol       - Tolerance for stopping criterion.
    % maxIter   - Maximum number of iterations.
    % epsilon   - Small perturbation for numerical gradient (default: 1e-6).
    %
    % Outputs:
    % grad      - Computed gradients.
    % fevals    - Counter of function evaluations.

    arguments
        func function_handle
        x (:, 1) double
        epsilon (1, 1) double = 1e-6
    end

    % Generate a random sign vector (+1 or -1 for each component)
    n = length(x);
    random_signs = sign(randn(n, 1));
    perturbations = epsilon * random_signs;

    % Apply perturbations to input vector in parallel
    x_0 = x;
    x_perturb = x + perturbations;

    % Evaluate function at perturbed points
    f_0 = func(x_0);
    f_perturb = func(x_perturb);

    % Compute the gradient for all components
    grad = (f_perturb - f_0) ./ (perturbations);
    fevals = 2;
end
