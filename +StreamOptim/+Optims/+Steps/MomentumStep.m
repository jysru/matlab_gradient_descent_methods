function [x_new, v] = MomentumStep(x, grad, v, opts)
    % Momentum step
    % 
    % Inputs:
    % x             - Previous variable vector.
    % v             - Velocity vector.
    % opts.alpha    - Learning rate (step size).
    % opts.beta     - Momentum factor
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


    % Update velocity
    v = opts.beta * v + (1 - opts.beta) * grad;
    
    % Update x using Momentum rule
    x_new = x - opts.alpha * v;
end
