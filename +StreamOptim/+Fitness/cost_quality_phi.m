function [cost] = cost_quality_phi(phi, phit, noise_std)
    cost =  1 - (StreamOptim.Fitness.quality_phi(phi, phit) - abs(noise_std * randn()));
end

