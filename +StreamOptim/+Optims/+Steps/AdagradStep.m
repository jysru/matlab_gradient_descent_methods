function [x_new, sum_sq_grad] = AdagradStep(x, grad, sum_sq_grad, opts)
    % Adagrad step
    % 
    % Inputs:
    % x             - Previous variable vector.
    % sum_sq_grad   - Sum of squared gradients.
    % E_dx_sq       - Running average average of squared parameter updates.
    % opts.alpha    - Learning rate (step size).
    % opts.epsilon  - Small constant to avoid division by zero.
    % 
    % Outputs:
    % x_new         - Updated variable vector.

    arguments
        x (:, 1) double
        grad (:, 1) double
        sum_sq_grad (:, 1) double
        opts.alpha (1, 1) double = 1e-3
        opts.epsilon (1, 1) double = 1e-9
    end

    % Update sum of squared gradients
    sum_sq_grad = sum_sq_grad + grad.^2;

    % Update x using Adagrad rule
    x_new = x - opts.alpha * grad ./ (sqrt(sum_sq_grad) + opts.epsilon);
end
