classdef Optimizer < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here

    properties (SetAccess = protected, GetAccess = public)
        history
        func
        x0
        x_opt
        alpha
        beta
        beta1
        beta2
        rho
        tol
        maxIter
        epsilon
        ub
        lb
        grad_func
        available_algorithms = {'GD', 'Momentum', 'NAG', 'Adam', 'NAdam', 'RAdam', 'Adamax', 'Adadelta', 'Adagrad', 'RMSProp', 'ADMM'}
        default_algorithm = 'GradientDescent'
        current_algorithm
        figure_handle
        axes_handle
        plot_handle
        plot_each_iter
    end


    methods (Access = public)
        function obj = Optimizer(func, x0, alpha, opts)
            arguments
                func function_handle
                x0 (:, 1) double
                alpha (1, 1) double = 1e-3 % Learning rate
                
                opts.beta (1, 1) double = 0.9 % Momentum factor for Momentum or Nesterov methods
                opts.beta1 (1, 1) double = 0.9 % Decay rate for the first moment estimates (Adamax or Adam-family methods)
                opts.beta2 (1, 1) double = 0.999 % Decay rate for the second moment estimates (Adamax or Adam-family methods)
                opts.rho (1, 1) double = 1.0 % Augmented Lagrangian parameter for ADMM
                opts.tol (1, 1) double = 0 
                opts.maxIter (1, 1) double = 100
                opts.epsilon (1, 1) double = 1e-6
                opts.ub (:, 1) double = []
                opts.lb (:, 1) double = []
                opts.grad_func function_handle = @StreamOptim.Gradients.CentralFiniteDifferences
            end
            
            obj.func = func;
            obj.x0 = x0;
            obj.alpha = alpha;
            obj.beta = opts.beta;
            obj.beta1 = opts.beta1;
            obj.beta2 = opts.beta2;
            obj.rho = opts.rho;
            obj.tol = opts.tol;
            obj.maxIter = opts.maxIter;
            obj.epsilon = opts.epsilon;
            obj.ub = opts.ub;
            obj.lb = opts.lb;
            obj.grad_func = opts.grad_func;
        end

        function Run(obj, opts)
            arguments
                obj
                opts.algorithm (1, :) char = 'GD'
                opts.plot_each_iter (1, 1) logical = false
            end

            obj.current_algorithm = obj.AlgorithmSelection(name=opts.algorithm);
            disp(['Selected optimization algorithm: ' obj.current_algorithm]);
            obj.plot_each_iter = opts.plot_each_iter;

            obj.(obj.current_algorithm)();
        end

        function PlotConvergence(obj, opts)
            arguments
                obj
                opts.FigureNumber (1, 1) double = 1
                opts.Marker (1, :) char = '.'
                opts.MarkerSize (1, 1) double = 15
                opts.LineStyle (1, :) char = 'none'
                opts.Color = 'b';
                opts.YScale (1, :) char = 'lin'
                opts.YLim = []
                opts.Reset (1, 1) logical = false
            end

            if isempty(obj.figure_handle) || ~isvalid(obj.figure_handle) || ~ishandle(obj.figure_handle)
                obj.InitPlot( ...
                    FigureNumber=opts.FigureNumber, ...
                    Marker=opts.Marker, ...
                    MarkerSize=opts.MarkerSize, ...
                    LineStyle=opts.LineStyle, ...
                    Color=opts.Color, ...
                    YScale=opts.YScale, ...
                    YLim=opts.YLim);
            else
                obj.plot_handle.YData = obj.history.fval;
                if ~isempty(opts.YLim)
                    ylim(opts.YLim)
                end
            end

        end
    end


    methods (Access = private)
        function InitPlot(obj, opts)
            arguments
                obj
                opts.FigureNumber (1, 1) double = 1
                opts.Marker (1, :) char = '.'
                opts.MarkerSize (1, 1) double = 15
                opts.LineStyle (1, :) char = 'none'
                opts.Color = 'b';
                opts.YScale (1, :) char = 'lin'
                opts.YLim = []
                opts.Reset (1, 1) logical = false
            end

            obj.figure_handle = figure(opts.FigureNumber); clf
            obj.axes_handle = gca();
            obj.plot_handle = plot(nan, 'Marker', opts.Marker, 'MarkerSize', opts.MarkerSize, 'LineStyle', opts.LineStyle);
            title([obj.current_algorithm ' optimization'])
            xlabel('Iteration #')
            ylabel('Cost function')
            grid on
            if ~isempty(opts.YLim)
                ylim(opts.YLim)
            end
            set(gca, 'YScale', opts.YScale)
        end

        
        function selection = AlgorithmSelection(obj, opts)
            arguments
                obj
                opts.name (1, :) char = 'GD'
            end

            switch lower(opts.name)
                case 'gd'
                    selection = 'GradientDescent';
                case 'gradientdescent'
                    selection = 'GradientDescent';
                case 'momentum'
                    selection = 'Momentum';
                case 'nag'
                    selection = 'NesterovAcceleratedGradient';
                case 'nesterov'
                    selection = 'NesterovAcceleratedGradient';
                case 'nesterovacceleratedgradient'
                    selection = 'NesterovAcceleratedGradient';
                case 'adam'
                    selection = 'Adam';
                case 'nadam'
                    selection = 'NAdam';
                case 'radam'
                    selection = 'RAdam';
                case 'adamax'
                    selection = 'Adamax';
                case 'adadelta'
                    selection = 'Adadelta';
                case 'adagrad'
                    selection = 'Adagrad';
                case 'rmsprop'
                    selection = 'RMSProp';
                case 'admm'
                    selection = 'ADMM';
                otherwise
                    disp(['Defaulting to ' obj.default_algorithm ' optimization!'])
                    selection = obj.default_algorithm;
            end
        end


        function GradientDescent(obj)
            % Initialize variables
            x = obj.x0;
            history.x = x;
            history.fval = obj.func(x);
            history.fevals = 0;
            history.diff = [];
            
            for iter = 1:obj.maxIter
                % Compute the numerical gradient and run a step
                [grad, fevals] = obj.grad_func(obj.func, x, obj.epsilon);
                x_new = StreamOptim.Optims.Steps.GradientDescentStep(x, grad, alpha=obj.alpha);
                
                % Compute the change in the variables
                diff = norm(x_new - x);
                
                % Update x and constrain within range
                x = x_new;
                x = StreamOptim.Utils.constrainWithinRange(x, obj.lb, obj.ub);
                obj.x_opt = x;
                
                % Store the history
                history.x = [history.x, x];
                history.fval = [history.fval, obj.func(obj.x_opt)];
                history.fevals = history.fevals + fevals;
                history.diff = [history.diff, diff];
                obj.history = history;

                % Plot convergence
                if obj.plot_each_iter
                    obj.PlotConvergence();
                end
                
                % Check the stopping criterion
                if diff < obj.tol
                    break;
                end
            end

            % Output the number of iterations
            fprintf('Gradient Descent converged in %d iterations.\n', iter);
        end


        function Momentum(obj)
            % Initialize variables
            x = obj.x0;
            history.x = x;
            history.fval = obj.func(x);
            history.fevals = 0;
            history.diff = [];

            % Initialize velocity
            v = zeros(length(x), 1); % Squared gradient vector
            
            for iter = 1:obj.maxIter
                % Compute the numerical gradient and run a step
                [grad, fevals] = obj.grad_func(obj.func, x, obj.epsilon);
                [x_new, v] = StreamOptim.Optims.Steps.MomentumStep(x, grad, v, alpha=obj.alpha, beta=obj.beta);
                
                % Compute the change in the variables
                diff = norm(x_new - x);
                
                % Update x and constrain within range
                x = x_new;
                x = StreamOptim.Utils.constrainWithinRange(x, obj.lb, obj.ub);
                obj.x_opt = x;
                
                % Store the history
                history.x = [history.x, x];
                history.fval = [history.fval, obj.func(obj.x_opt)];
                history.fevals = history.fevals + fevals;
                history.diff = [history.diff, diff];
                obj.history = history;

                % Plot convergence
                if obj.plot_each_iter
                    obj.PlotConvergence();
                end
                
                % Check the stopping criterion
                if diff < obj.tol
                    break;
                end
            end
            
            % Output the number of iterations
            fprintf('Momentum converged in %d iterations.\n', iter);
        end


        function NesterovAcceleratedGradient(obj)
            % Initialize variables
            x = obj.x0;
            history.x = x;
            history.fval = obj.func(x);
            history.fevals = 0;
            history.diff = [];

            % Initialize velocity
            v = zeros(length(x), 1); % Squared gradient vector
            
            for iter = 1:obj.maxIter
                % Lookahead step
                lookahead_x = x - obj.beta * v;

                % Compute the numerical gradient
                [grad, fevals] = obj.grad_func(obj.func, lookahead_x, obj.epsilon);
        
                % Update the variables using the gradient descent step
                [x_new, v] = StreamOptim.Optims.Steps.NesterovAcceleratedGradientStep(x, grad, v, alpha=obj.alpha, beta=obj.beta);
                
                % Compute the change in the variables
                diff = norm(x_new - x);

                % Update x and constrain within range
                x = x_new;
                x = StreamOptim.Utils.constrainWithinRange(x, obj.lb, obj.ub);
                obj.x_opt = x;
                
                % Store the history
                history.x = [history.x, x];
                history.fval = [history.fval, obj.func(obj.x_opt)];
                history.fevals = history.fevals + fevals;
                history.diff = [history.diff, diff];
                obj.history = history;

                % Plot convergence
                if obj.plot_each_iter
                    obj.PlotConvergence();
                end
                
                % Check the stopping criterion
                if diff < obj.tol
                    break;
                end
            end
            
            % Output the number of iterations
            fprintf('Nesterov Accelerated Gradient converged in %d iterations.\n', iter);
        end


        function Adam(obj)
            % Initialize variables
            x = obj.x0;
            history.x = x;
            history.fval = obj.func(x);
            history.fevals = 0;
            history.diff = [];
        
            % Initialize the first and second moment vectors
            m = zeros(length(x), 1); % First moment vector
            v = zeros(length(x), 1); % Second moment vector
            
            for iter = 1:obj.maxIter
                % Compute the numerical gradient
                [grad, fevals] = obj.grad_func(obj.func, x, obj.epsilon);
        
                % Update the variables using the gradient descent step
                [x_new, m, v] = StreamOptim.Optims.Steps.AdamStep(x, grad, m, v, iter, alpha=obj.alpha, beta1=obj.beta1, beta2=obj.beta2);
                
                % Compute the change in the variables
                diff = norm(x_new - x);

                % Update x and constrain within range
                x = x_new;
                x = StreamOptim.Utils.constrainWithinRange(x, obj.lb, obj.ub);
                obj.x_opt = x;
                
                % Store the history
                history.x = [history.x, x];
                history.fval = [history.fval, obj.func(obj.x_opt)];
                history.fevals = history.fevals + fevals;
                history.diff = [history.diff, diff];
                obj.history = history;

                % Plot convergence
                if obj.plot_each_iter
                    obj.PlotConvergence();
                end
                
                % Check the stopping criterion
                if diff < obj.tol
                    break;
                end
            end
            
            % Output the number of iterations
            fprintf('Adam converged in %d iterations.\n', iter);
        end


        function NAdam(obj)
            % Initialize variables
            x = obj.x0;
            history.x = x;
            history.fval = obj.func(x);
            history.fevals = 0;
            history.diff = [];
        
            % Initialize the first and second moment vectors
            m = zeros(length(x), 1); % First moment vector
            v = zeros(length(x), 1); % Second moment vector
            
            for iter = 1:obj.maxIter
                % Compute the numerical gradient
                [grad, fevals] = obj.grad_func(obj.func, x, obj.epsilon);
        
                % Update the variables using the gradient descent step
                [x_new, m, v] = StreamOptim.Optims.Steps.NAdamStep(x, grad, m, v, iter, alpha=obj.alpha, beta1=obj.beta1, beta2=obj.beta2);
                
                % Compute the change in the variables
                diff = norm(x_new - x);
                
                % Update x and constrain within range
                x = x_new;
                x = StreamOptim.Utils.constrainWithinRange(x, obj.lb, obj.ub);
                obj.x_opt = x;
                
                % Store the history
                history.x = [history.x, x];
                history.fval = [history.fval, obj.func(obj.x_opt)];
                history.fevals = history.fevals + fevals;
                history.diff = [history.diff, diff];
                obj.history = history;

                % Plot convergence
                if obj.plot_each_iter
                    obj.PlotConvergence();
                end
                
                % Check the stopping criterion
                if diff < obj.tol
                    break;
                end
            end
            
            % Output the number of iterations
            fprintf('NAdam converged in %d iterations.\n', iter);
        end


        function RAdam(obj)
            % Initialize variables
            x = obj.x0;
            history.x = x;
            history.fval = obj.func(x);
            history.fevals = 0;
            history.diff = [];
        
            % Initialize the first and second moment vectors
            m = zeros(length(x), 1); % First moment vector
            v = zeros(length(x), 1); % Second moment vector
            rho_inf = 2 / (1 - obj.beta2) - 1; % rho_inf as defined in RAdam paper
            
            for iter = 1:obj.maxIter
                % Compute the numerical gradient
                [grad, fevals] = obj.grad_func(obj.func, x, obj.epsilon);
        
                % Update the variables using the gradient descent step
                [x_new, m, v, rho_inf] = StreamOptim.Optims.Steps.RAdamStep(x, grad, m, v, rho_inf, iter, alpha=obj.alpha, beta1=obj.beta1, beta2=obj.beta2);
                
                % Compute the change in the variables
                diff = norm(x_new - x);
                
                % Update x and constrain within range
                x = x_new;
                x = StreamOptim.Utils.constrainWithinRange(x, obj.lb, obj.ub);
                obj.x_opt = x;
                
                % Store the history
                history.x = [history.x, x];
                history.fval = [history.fval, obj.func(obj.x_opt)];
                history.fevals = history.fevals + fevals;
                history.diff = [history.diff, diff];
                obj.history = history;

                % Plot convergence
                if obj.plot_each_iter
                    obj.PlotConvergence();
                end
                
                % Check the stopping criterion
                if diff < obj.tol
                    break;
                end
            end
            
            % Output the number of iterations
            fprintf('RAdam converged in %d iterations.\n', iter);
        end


        function Adamax(obj)
            % Initialize variables
            x = obj.x0;
            history.x = x;
            history.fval = obj.func(x);
            history.fevals = 0;
            history.diff = [];
        
            % Initialize the first and second moment vectors
            m = zeros(length(x), 1); % First moment vector
            u = zeros(length(x), 1); % Infinity norm of gradients
            
            for iter = 1:obj.maxIter
                % Compute the numerical gradient
                [grad, fevals] = obj.grad_func(obj.func, x, obj.epsilon);
        
                % Update the variables using the gradient descent step
                [x_new, m, u] = StreamOptim.Optims.Steps.AdamaxStep(x, grad, m, u, iter, alpha=obj.alpha, beta1=obj.beta1, beta2=obj.beta2);
                
                % Compute the change in the variables
                diff = norm(x_new - x);
                
                % Update x and constrain within range
                x = x_new;
                x = StreamOptim.Utils.constrainWithinRange(x, obj.lb, obj.ub);
                obj.x_opt = x;
                
                % Store the history
                history.x = [history.x, x];
                history.fval = [history.fval, obj.func(obj.x_opt)];
                history.fevals = history.fevals + fevals;
                history.diff = [history.diff, diff];
                obj.history = history;

                % Plot convergence
                if obj.plot_each_iter
                    obj.PlotConvergence();
                end
                
                % Check the stopping criterion
                if diff < obj.tol
                    break;
                end
            end
            
            % Output the number of iterations
            fprintf('Adamax converged in %d iterations.\n', iter);
        end


        function Adadelta(obj)
            % Initialize variables
            x = obj.x0;
            history.x = x;
            history.fval = obj.func(x);
            history.fevals = 0;
            history.diff = [];
        
            % Initialize the first and second moment vectors
            E_grad_sq = zeros(length(x), 1); % Initialize running average of squared gradients
            E_dx_sq = 1e-3 * ones(length(x), 1); % Initialize running average of squared parameter updates, not to 0 or it stagnates
            
            for iter = 1:obj.maxIter
                % Compute the numerical gradient
                [grad, fevals] = obj.grad_func(obj.func, x, obj.epsilon);

                % Update the variables using the gradient descent step
                [x_new, E_grad_sq, E_dx_sq] = StreamOptim.Optims.Steps.AdadeltaStep(x, grad, E_grad_sq, E_dx_sq, beta=obj.beta);

                % Compute the change in the variables
                diff = norm(x_new - x);
                
                % Update x and constrain within range
                x = x_new;
                x = StreamOptim.Utils.constrainWithinRange(x, obj.lb, obj.ub);
                obj.x_opt = x;
                
                % Store the history
                history.x = [history.x, x];
                history.fval = [history.fval, obj.func(obj.x_opt)];
                history.fevals = history.fevals + fevals;
                history.diff = [history.diff, diff];
                obj.history = history;

                % Plot convergence
                if obj.plot_each_iter
                    obj.PlotConvergence();
                end
                
                % Check the stopping criterion
                if diff < obj.tol
                    break;
                end
            end
            
            % Output the number of iterations
            fprintf('Adadelta converged in %d iterations.\n', iter);
        end


        function Adagrad(obj)
            % Initialize variables
            x = obj.x0;
            history.x = x;
            history.fval = obj.func(x);
            history.fevals = 0;
            history.diff = [];
        
            % Initialize sum of squared gradients
            sum_sq_grad = 1e-6 * ones(length(x), 1);

            for iter = 1:obj.maxIter
                % Compute the numerical gradient
                [grad, fevals] = obj.grad_func(obj.func, x, obj.epsilon);

                % Update the variables using the gradient descent step
                [x_new, sum_sq_grad] = StreamOptim.Optims.Steps.AdagradStep(x, grad, sum_sq_grad, alpha=obj.alpha);

                % Compute the change in the variables
                diff = norm(x_new - x);
                
                % Update x and constrain within range
                x = x_new;
                x = StreamOptim.Utils.constrainWithinRange(x, obj.lb, obj.ub);
                obj.x_opt = x;
                
                % Store the history
                history.x = [history.x, x];
                history.fval = [history.fval, obj.func(obj.x_opt)];
                history.fevals = history.fevals + fevals;
                history.diff = [history.diff, diff];
                obj.history = history;

                % Plot convergence
                if obj.plot_each_iter
                    obj.PlotConvergence();
                end
                
                % Check the stopping criterion
                if diff < obj.tol
                    break;
                end
            end
            
            % Output the number of iterations
            fprintf('Adagrad converged in %d iterations.\n', iter);
        end


        function RMSProp(obj)
            % Initialize variables
            x = obj.x0;
            history.x = x;
            history.fval = obj.func(x);
            history.fevals = 0;
            history.diff = [];
        
            % Initialize the moving average of squared gradients
            v = zeros(length(x), 1); % Squared gradient vector
            
            for iter = 1:obj.maxIter
                % Compute the numerical gradient
                [grad, fevals] = obj.grad_func(obj.func, x, obj.epsilon);
        
                % Update the variables using the gradient descent step
                [x_new, v] = StreamOptim.Optims.Steps.RMSPropStep(x, grad, v, alpha=obj.alpha, beta=obj.beta);
                
                % Compute the change in the variables
                diff = norm(x_new - x);
                
                % Update x and constrain within range
                x = x_new;
                x = StreamOptim.Utils.constrainWithinRange(x, obj.lb, obj.ub);
                obj.x_opt = x;
                
                % Store the history
                history.x = [history.x, x];
                history.fval = [history.fval, obj.func(obj.x_opt)];
                history.fevals = history.fevals + fevals;
                history.diff = [history.diff, diff];
                obj.history = history;

                % Plot convergence
                if obj.plot_each_iter
                    obj.PlotConvergence();
                end
                
                % Check the stopping criterion
                if diff < obj.tol
                    break;
                end
            end
            
            % Output the number of iterations
            fprintf('RMSProp converged in %d iterations.\n', iter);
        end


        function ADMM(obj)
            % Initialize variables
            x = obj.x0;
            history.x = x;
            history.fval = obj.func(x);
            history.fevals = 0;
            history.diff = [];
        
            % Initialize variables
            z = x; % Auxiliary variable
            u = zeros(length(x), 1); % Dual variable
            
            for iter = 1:obj.maxIter
                % Compute the numerical gradient
                [grad, fevals] = obj.grad_func(obj.func, x, obj.epsilon);

                % Update the variables using the ADMM step
                [x_new, z, u] = StreamOptim.Optims.Steps.ADMMStep(x, grad, z, u, alpha=obj.alpha, rho=obj.rho);
        
                % Compute the change in the variables
                diff = norm(x_new - x);
                
                % Update x
                x = x_new;
                x = StreamOptim.Utils.constrainWithinRange(x, obj.lb, obj.ub);
                obj.x_opt = x;
                
                % z-update: Projection step onto [0, 2*pi]
                z_old = z;
                z = x + u;
                z = StreamOptim.Utils.constrainWithinRange(z, obj.lb, obj.ub);
        
                % u-update: Dual variable update
                u = u + (x - z);
                u = StreamOptim.Utils.constrainWithinRange(u, obj.lb, obj.ub);
                
                % Store the history
                history.x = [history.x, x];
                history.fval = [history.fval, obj.func(obj.x_opt)];
                history.fevals = history.fevals + fevals;
                history.diff = [history.diff, diff];
                obj.history = history;

                % Plot convergence
                if obj.plot_each_iter
                    obj.PlotConvergence();
                end
                
                % Check the stopping criterion
                if diff < obj.tol
                    break;
                end
            end
            
            % Output the number of iterations
            fprintf('ADMM converged in %d iterations.\n', iter);
        end

    end
end

