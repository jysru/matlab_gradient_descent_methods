function [x_new, v] = RMSPropStep(x, grad, v, opts)
    % RMSProp step
    % 
    % Inputs:
    % x             - Previous variable vector.
    % v             - Moving average of squared gradients vector.
    % opts.alpha    - Learning rate (step size).
    % opts.beta     - Exponential decay rate for the squared gradients.
    % opts.epsilon  - Small constant to avoid division by zero.
    % 
    % Outputs:
    % x_new     - Updated variable vector.

    arguments
        x (:, 1) double
        grad (:, 1) double
        v (:, 1) double
        opts.alpha (1, 1) double = 1e-3
        opts.beta (1, 1) double = 0.9
        opts.epsilon (1, 1) double = 1e-9
    end

    % Update the moving average of squared gradients
    v = opts.beta * v + (1 - opts.beta) * (grad .^ 2);
    
    % Update x using RMSprop rule
    x_new = x - opts.alpha * (grad ./ (sqrt(v) + opts.epsilon));
end
