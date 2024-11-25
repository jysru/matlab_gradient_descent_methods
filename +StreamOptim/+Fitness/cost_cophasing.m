function qphi = cost_cophasing(phi_vector)
    qphi = 1 - StreamOptim.Fitness.cophasing(phi_vector);
end

