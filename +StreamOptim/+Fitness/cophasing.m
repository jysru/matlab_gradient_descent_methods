function qphi = cophasing(phi_vector)
    x = ones(size(phi_vector)) .* exp(1j * phi_vector);
    xt = ones(size(phi_vector));
    qphi = StreamOptim.Fitness.quality(x, xt);
    % qphi = sum(qphi, "all");
end

