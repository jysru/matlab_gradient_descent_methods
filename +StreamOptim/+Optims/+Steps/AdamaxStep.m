function [x_new, m, u, step] = AdamaxStep(x, grad, m, u, iter, opts)
    % Adamax step
    % 
    % Inputs:
    % x             - Previous variable vector.
    % m             - First moments vector.
    % u             - Infinity norm of gradients.
    % opts.alpha    - Learning rate (step size).
    % opts.beta1    - Exponential decay rate for the first moment estimates.
    % opts.beta2    - Exponential decay rate for the second moment estimates.
    % opts.epsilon  - Small constant to avoid division by zero.
    % 
    % Outputs:
    % x_new     - Updated variable vector.

    arguments
        x (:, 1) double
        grad (:, 1) double
        m (:, 1) double
        u (:, 1) double
        iter (1, 1) double
        opts.alpha (1, 1) double = 1e-3
        opts.beta1 (1, 1) double = 0.9
        opts.beta2 (1, 1) double = 0.999
        opts.epsilon (1, 1) double = 1e-9
    end

    % Update biased first moment estimate
    m = opts.beta1 * m + (1 - opts.beta1) * grad;

    % Update infinity norm
    u = max(opts.beta2 * u, abs(grad));
    
    % Compute bias-corrected first moment estimate
    m_hat = m / (1 - opts.beta1 ^ iter);

    % Compute the step to be applied
    step = m_hat ./ (u + opts.epsilon);
    
    % Update x using Adamax rule
    x_new = x - opts.alpha * step;
end
