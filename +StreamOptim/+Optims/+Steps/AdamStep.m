function [x_new, m, v] = AdamStep(x, grad, m, v, iter, opts)
    % Adam step
    % 
    % Inputs:
    % x             - Previous variable vector.
    % m             - First moments vector.
    % v             - Second moments vector.
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
        v (:, 1) double
        iter (1, 1) double
        opts.alpha (1, 1) double = 1e-3
        opts.beta1 (1, 1) double = 0.9
        opts.beta2 (1, 1) double = 0.999
        opts.epsilon (1, 1) double = 1e-9
    end

    % Update biased first moment estimate
    m = opts.beta1 * m + (1 - opts.beta1) * grad;
    
    % Update biased second raw moment estimate
    v = opts.beta2 * v + (1 - opts.beta2) * (grad .^ 2);
    
    % Compute bias-corrected first moment estimate
    m_hat = m / (1 - opts.beta1 ^ iter);
    
    % Compute bias-corrected second raw moment estimate
    v_hat = v / (1 - opts.beta2 ^ iter);
    
    % Update phi using Adam rule
    x_new = x - opts.alpha * (m_hat ./ (sqrt(v_hat) + opts.epsilon));
end
