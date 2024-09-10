function [grad, fevals, f_plus, f_minus] = ParallelRandomSignCentralFiniteDifferences(func, x, epsilon)
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
    x_plus = x + perturbations;
    x_minus = x - perturbations;

    % Evaluate function at perturbed points
    f_plus = func(x_plus);
    f_minus = func(x_minus);

    % Compute the gradient for all components
    grad = (f_plus - f_minus) ./ (2 * perturbations);
    fevals = 2;
end
