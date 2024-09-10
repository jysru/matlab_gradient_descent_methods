function qphi = quality_phi(phi_vector, phi_target)
    x = ones(size(phi_vector)) .* exp(1j * phi_vector);
    xt = ones(size(phi_target)) .* exp(1j * phi_target);
    qphi = StreamOptim.Fitness.quality(x, xt);
    qphi = sum(qphi, "all");
end

