function [grad, fevals, f_plus, f_minus] = RandomSignCentralFiniteDifferences(func, x, epsilon)
    % Compute the numerical gradient using central finite differences with random signs
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
    grad = zeros(n, 1);
    f_plus = zeros(n, 1);
    f_minus = zeros(n, 1);
    
    fevals = 0;
    for i = 1:n
        % Compute the numerical gradient in a random direction for the i-th component
        perturbation = epsilon * random_signs(i);

        % Reset perturbated input vector
        x_plus = x;
        x_minus = x;

        % Apply perturbation to looped variable
        x_plus(i) = x(i) + perturbation;
        x_minus(i) = x(i) - perturbation;

        % Evaluate the cost function for the looped variable
        f_p = func(x_plus);
        f_m = func(x_minus);

        % Store values for outputs
        f_plus(i) = f_p;
        f_minus(i) = f_m;
        
        % Compute the gradient for the looped variable
        grad(i) = (f_p - f_m) / (2 * perturbation);

        % Increment the function evaluation counter
        fevals = fevals + 2;
    end
end
