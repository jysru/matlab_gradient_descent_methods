function [x_opt, fval, history] = ADMM(func, x0, alpha, opts)
    % ADMM optimization
    % 
    % Inputs:
    % func              - Handle to the objective function to minimize.
    % x0                - Initial vector guess for the variables.
    % alpha             - Learning rate (step size).
    % opts.rho          - Augmented Lagrangian parameter
    % opts.tol          - Tolerance for stopping criterion (if 0, it is going to run to maxIter).
    % opts.maxIter      - Maximum number of iterations.
    % opts.epsilon      - Small perturbation for numerical gradient (default: 1e-6).
    % opts.ub           - Upper boundaries vector.
    % opts.lb           - Lower boundaries vector.
    % opts.grad_func    - Handle to the gradient evaluation function
    % 
    % Outputs:
    % x_opt     - The optimized variable.
    % fval      - The value of the objective function at x_opt.
    % history   - A structure to store the history of the optimization.

    arguments
        func function_handle
        x0 (:, 1) double
        alpha (1, 1) double = 1e-3
        opts.rho (1, 1) double = 1.0
        opts.tol (1, 1) double = 0
        opts.maxIter (1, 1) double = 100
        opts.epsilon (1, 1) double = 1e-6
        opts.ub (:, 1) double = []
        opts.lb (:, 1) double = []
        opts.grad_func function_handle = @StreamOptim.Gradients.CentralFiniteDifferences
    end
    
    % Initialize variables
    x = x0;
    history.x = x;
    history.fval = func(x);
    history.fevals = 0;
    history.diff = [];

    % Initialize variables
    z = x; % Auxiliary variable
    u = zeros(length(x), 1); % Dual variable
    
    for iter = 1:opts.maxIter
        % Compute the numerical gradient
        [grad, fevals] = opts.grad_func(func, x, opts.epsilon);

        % Update the variables using the ADMM step
        [x_new] = StreamOptim.Optims.Steps.ADMMStep(x, grad, z, u, alpha=alpha, rho=opts.rho);
        
        % Compute the change in the variables
        diff = norm(x_new - x);
        
        % Update x
        x = x_new;
        x = StreamOptim.Utils.constrainWithinRange(x, opts.lb, opts.ub);

        % z-update: Projection step onto [0, 2*pi]
        z_old = z;
        z = x + u;
        z = StreamOptim.Utils.constrainWithinRange(z, opts.lb, opts.ub);

        % u-update: Dual variable update
        u = u + (x - z);
        u = StreamOptim.Utils.constrainWithinRange(u, opts.lb, opts.ub);
        
        % Store the history
        history.x = [history.x, x];
        history.fval = [history.fval, func(x)];
        history.fevals = history.fevals + fevals;
        history.diff = [history.diff, diff];
        history.options = opts;
        
        % Check the stopping criterion
        if diff < opts.tol
            break;
        end
    end
    
    % Return the optimized variable and its function value
    x_opt = x;
    fval = func(x_opt);
    
    % Output the number of iterations
    fprintf('ADMM converged in %d iterations.\n', iter);
end