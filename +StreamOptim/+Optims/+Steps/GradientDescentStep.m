function x_new = GradientDescentStep(x, grad, opts)
    % Gradient Descent step
    % 
    % Inputs:
    % x         - Previous variable vector.
    % grad      - Evaluated gradients vector.
    % alpha     - Learning rate (step size).
    % 
    % Outputs:
    % x_new     - Updated variable vector.

    arguments
        x (:, 1) double
        grad (:, 1) double
        opts.alpha (1, 1) double = 1e-3
    end

    % Update x using Gradient Descent rule
    x_new = x - opts.alpha * grad;
end
