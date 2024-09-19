function [x_new, z, u, step] = ADMMStep(x, grad, z, u, opts)
    % ADMM step
    % 
    % Inputs:
    % x             - Previous variable vector.
    % z             - Auxiliary variable vector.
    % u             - Dual variable vector.
    % opts.alpha    - Learning rate (step size).
    % opts.rho      - Augmented Lagrangian parameter
    % opts.epsilon  - Small constant to avoid division by zero.
    % 
    % Outputs:
    % x_new     - Updated variable vector.

    arguments
        x (:, 1) double
        grad (:, 1) double
        z (:, 1) double
        u (:, 1) double
        opts.alpha (1, 1) double = 1e-3
        opts.rho (1, 1) double = 0.9
        opts.epsilon (1, 1) double = 1e-9
    end

    % x-update: Minimize f(phi) + (rho/2) * norm(phi - z + u)^2
    % Using a gradient descent step for simplicity
    grad = grad + opts.rho * (x - z + u);

    % Compute the step to be applied
    step = grad / (opts.rho + 1);

    % Update step with learning rate adjusted by rh
    x_new = x - step; 
end
