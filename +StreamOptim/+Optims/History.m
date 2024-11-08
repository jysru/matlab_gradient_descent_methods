classdef History < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here

    properties (SetAccess = protected, GetAccess = public)
        algorithm
        x
        fvals
        fevals
        diffs
        steps
        alpha
        grads
        fvals1
        fvals2

        fvals_figure_handle
        fvals_axes_handle
        fvals_plot_handle

        diffs_figure_handle
        diffs_axes_handle
        diffs_plot_handle

        grads_figure_handle
        grads_axes_handle
        grads_plot_handle

        noise_figure_handle
        steps_figure_handle
    end

    methods (Access = public)
        function obj = History(algorithm, x, fvals)
            arguments
                algorithm (1, :) char
                x (:, 1) double
                fvals (:, 1) double
            end
            obj.algorithm = algorithm;
            obj.x = x;
            obj.fvals = fvals;
            obj.fevals = 0;
            obj.alpha = [];
            obj.steps = [];
            obj.diffs = [];
            obj.grads = [];
            obj.fvals1 = [];
            obj.fvals2 = [];
        end

        function Update(obj, opts)
            arguments
                obj
                opts.x (:, 1) double = []
                opts.fvals (:, 1) double = []
                opts.fevals (1, 1) double = []
                opts.alpha (1, 1) double = []
                opts.diffs (:, 1) double = []
                opts.steps (:, 1) double = []
                opts.grads (:, 1) double = []
                opts.fvals1 (:, 1) double = []
                opts.fvals2 (:, 1) double = []
            end
            obj.x = [obj.x, opts.x];
            obj.fvals = [obj.fvals, opts.fvals];
            obj.fevals = obj.fevals + opts.fevals;
            obj.alpha = [obj.alpha, opts.alpha];
            obj.diffs = [obj.diffs, opts.diffs];
            obj.steps = [obj.steps, opts.steps];
            obj.grads = [obj.grads, opts.grads];
            obj.fvals1 = [obj.fvals1, opts.fvals1];
            obj.fvals2 = [obj.fvals2, opts.fvals2];
        end

        function best = GetBest(obj)
            [~, idx] = min(obj.fvals);
            best.algorithm = obj.algorithm;
            best.x = obj.x(:, idx);
            best.fvals = obj.fvals(:, idx);
            best.diffs = obj.diffs(:, idx-1);
            best.steps = obj.steps(:, idx-1);
            best.alpha = obj.alpha(:, idx-1);
            best.grads = obj.grads(:, idx-1);
            best.fvals1 = obj.fvals1(:, idx-1);
            best.fvals2 = obj.fvals2(:, idx-1);
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

            if isempty(obj.fvals_figure_handle) || ~isvalid(obj.fvals_figure_handle) || ~ishandle(obj.fvals_figure_handle)
                obj.InitConvergencePlot( ...
                    FigureNumber=opts.FigureNumber, ...
                    Marker=opts.Marker, ...
                    MarkerSize=opts.MarkerSize, ...
                    LineStyle=opts.LineStyle, ...
                    Color=opts.Color, ...
                    YScale=opts.YScale, ...
                    YLim=opts.YLim);
            else
                obj.fvals_plot_handle.YData = obj.fvals;
                if ~isempty(opts.YLim)
                    ylim(opts.YLim)
                end
            end
        end

        function PlotSteps(obj, opts)
            arguments
                obj
                opts.FigureNumber (1, 1) double = 6
                opts.Marker (1, :) char = 'none'
                opts.MarkerSize (1, 1) double = 15
                opts.LineStyle (1, :) char = '-'
                opts.Color = 'r';
                opts.YScale (1, :) char = 'lin'
                opts.YLim = []
                opts.Reset (1, 1) logical = false
            end
            
            obj.noise_figure_handle = figure(opts.FigureNumber); clf; hold on; grid on; box on;
            mean_steps = mean(obj.steps, 1);
            std_steps = std(obj.steps, 0, 1);
            xaxis = linspace(1, length(mean_steps));
            max_steps = max(obj.steps, [], 1);
            min_steps = min(obj.steps, [], 1);
            
            fill([xaxis, fliplr(xaxis)], [mean_steps + std_steps, fliplr(mean_steps - std_steps)], 'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
            plot(max_steps, 'Marker', opts.Marker, 'MarkerSize', opts.MarkerSize, 'LineStyle', opts.LineStyle, 'LineWidth', 1, 'Color', 'g');
            plot(min_steps, 'Marker', opts.Marker, 'MarkerSize', opts.MarkerSize, 'LineStyle', opts.LineStyle, 'LineWidth', 1, 'Color', 'm');
            plot(mean_steps, 'Marker', opts.Marker, 'MarkerSize', opts.MarkerSize, 'LineStyle', opts.LineStyle, 'LineWidth', 1.5, 'Color', opts.Color);

            xlabel('Iteration #')
            ylabel('Input steps')
            title('Input steps evolution')
            

            legend('\mu \pm \sigma', 'max', 'min', '\mu')
        end

        function PlotNoiseWithPerturbs(obj, opts)
            arguments
                obj
                opts.FigureNumber (1, 1) double = 4
                opts.Marker (1, :) char = 'none'
                opts.MarkerSize (1, 1) double = 15
                opts.LineStyle (1, :) char = '-'
                opts.Color = 'b';
                opts.YScale (1, :) char = 'lin'
                opts.YLim = []
                opts.Reset (1, 1) logical = false
            end

            obj.noise_figure_handle = figure(opts.FigureNumber); clf;
            obj.noise_figure_handle.Position(3) = 1000;
            
            subplot(1, 2, 1); hold on; grid on; box on;
                plot(obj.fvals, 'Marker', opts.Marker, 'MarkerSize', opts.MarkerSize, 'LineStyle', opts.LineStyle, 'LineWidth', 1.5);
                plot(obj.fvals1, 'Marker', '.', 'MarkerSize', opts.MarkerSize, 'LineStyle', 'none');
                plot(obj.fvals2, 'Marker', '.', 'MarkerSize', opts.MarkerSize, 'LineStyle', 'none');
                title('Noise evaluation')
                xlabel('Iteration #')
                ylabel('Cost function')
                if ~isempty(opts.YLim)
                    ylim(opts.YLim)
                end
                xlim([0, length(obj.fvals)])
                set(gca, 'YScale', opts.YScale)
                legend('w/o perturb', 'w/ perturb 1', 'w/ perturb 2')

            subplot(1, 2, 2); hold on; grid on; box on;
                histogram(obj.fvals(:), EdgeAlpha=0, FaceAlpha=0.5)
                histogram(obj.fvals1(:), EdgeAlpha=0, FaceAlpha=0.3)
                histogram(obj.fvals2(:), EdgeAlpha=0, FaceAlpha=0.3)
                xlabel('Cost function value')
                ylabel('Density')
                title('Noisy cost function distributions')
                legend('w/o perturb', 'w/ perturb 1', 'w/ perturb 2')
        end

        function PlotDiffs(obj, opts)
            arguments
                obj
                opts.FigureNumber (1, 1) double = 2
                opts.Marker (1, :) char = 'none'
                opts.MarkerSize (1, 1) double = 15
                opts.LineStyle (1, :) char = '-'
                opts.Color = 'b';
                opts.YScale (1, :) char = 'lin'
                opts.YLim = []
                opts.Reset (1, 1) logical = false
            end

            obj.diffs_figure_handle = figure(opts.FigureNumber); clf
            obj.diffs_axes_handle = gca();
            obj.diffs_plot_handle = plot(obj.diffs, 'Marker', opts.Marker, 'MarkerSize', opts.MarkerSize, 'LineStyle', opts.LineStyle);
            title([obj.algorithm ' optimization: Norm of diffs'])
            xlabel('Iteration #')
            ylabel('Norm of variables diffs')
            grid on, box on
            if ~isempty(opts.YLim)
                ylim(opts.YLim)
            end
            set(gca, 'YScale', opts.YScale)
        end

        function PlotGrads(obj, opts)
            arguments
                obj
                opts.FigureNumber (1, 1) double = 3
                opts.Marker (1, :) char = 'none'
                opts.MarkerSize (1, 1) double = 15
                opts.LineStyle (1, :) char = '-'
                opts.Color = 'b';
                opts.YScale (1, :) char = 'lin'
                opts.YLim = []
                opts.Reset (1, 1) logical = false
            end

            obj.grads_figure_handle = figure(opts.FigureNumber); clf, hold on
            obj.grads_axes_handle = gca();
            for i=1:size(obj.x, 1)
                obj.grads_plot_handle(i) = plot(obj.grads(i, :), 'Marker', opts.Marker, 'MarkerSize', opts.MarkerSize, 'LineStyle', opts.LineStyle);
            end
            title([obj.algorithm ' optimization: Gradients'])
            xlabel('Iteration #')
            ylabel('Gradients')
            grid on, box on
            if ~isempty(opts.YLim)
                ylim(opts.YLim)
            end
            set(gca, 'YScale', opts.YScale)
        end
    end

    methods (Access = private)
        function InitConvergencePlot(obj, opts)
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

            obj.fvals_figure_handle = figure(opts.FigureNumber); clf
            obj.fvals_axes_handle = gca();
            obj.fvals_plot_handle = plot(nan, 'Marker', opts.Marker, 'MarkerSize', opts.MarkerSize, 'LineStyle', opts.LineStyle);
            title([obj.algorithm ' optimization: Convergence'])
            xlabel('Iteration #')
            ylabel('Cost function')
            grid on, box on
            if ~isempty(opts.YLim)
                ylim(opts.YLim)
            end
            set(gca, 'YScale', opts.YScale)
        end
    end
end


