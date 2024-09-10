function x_new = constrainWithinRange(x, lb, ub)
    % Constrain a vector within 
    % 
    % Inputs:
    % x         - Initial variable vector.
    % lb        - Lower boundaries vector or scalar.
    % ub        - Lower boundaries vector or scalar.
    % 
    % Outputs:
    % x_new     - Updated variable vector.

    arguments
        x (:, 1) double
        lb (:, 1) double = []
        ub (:, 1) double = []
    end

    x_new = x;
    if ~isempty(lb)
        x_new = max(x_new, lb);
    end
    if ~isempty(ub)
        x_new = min(x_new, ub);
    end
end

