function [x_new, E_grad_sq, E_dx_sq] = AdadeltaStep(x, grad, E_grad_sq, E_dx_sq, opts)
    % Adadelta step
    % 
    % Inputs:
    % x             - Previous variable vector.
    % E_grad_sq     - Running average of squared gradients.
    % E_dx_sq       - Running average average of squared parameter updates.
    % opts.beta     - Decay rate
    % opts.epsilon  - Small constant to avoid division by zero.
    % 
    % Outputs:
    % x_new         - Updated variable vector.

    arguments
        x (:, 1) double
        grad (:, 1) double
        E_grad_sq (:, 1) double
        E_dx_sq (:, 1) double
        opts.beta (1, 1) double = 0.95
        opts.epsilon (1, 1) double = 1e-9
    end

    % Update running average of squared gradients
    E_grad_sq = opts.beta * E_grad_sq + (1 - opts.beta) * grad.^2;

    % Compute parameter update
    RMS_dx = sqrt(E_dx_sq + opts.epsilon);
    RMS_grad = sqrt(E_grad_sq + opts.epsilon);
    delta_x = -(RMS_dx ./ RMS_grad) .* grad;

    % Update x using Adadelta rule
    x_new = x + delta_x;
    
    % Update running average of squared parameter updates
    E_dx_sq = opts.beta * E_dx_sq + (1 - opts.beta) * delta_x.^2;
end
