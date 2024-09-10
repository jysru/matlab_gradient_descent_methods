classdef History < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here

    properties (SetAccess = protected, GetAccess = public)
        algorithm
        x
        fvals
        fevals
        diffs
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
                opts.diffs (:, 1) double = []
                opts.grads (:, 1) double = []
                opts.fvals1 (:, 1) double = []
                opts.fvals2 (:, 1) double = []
            end
            obj.x = [obj.x, opts.x];
            obj.fvals = [obj.fvals, opts.fvals];
            obj.fevals = obj.fevals + opts.fevals;
            obj.diffs = [obj.diffs, opts.diffs];
            obj.grads = [obj.grads, opts.grads];
            obj.fvals1 = [obj.fvals1, opts.fvals1];
            obj.fvals2 = [obj.fvals2, opts.fvals2];
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

