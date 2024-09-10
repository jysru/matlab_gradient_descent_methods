function q = quality(vector, target)
    numel = abs(sum(vector .* conj(target)));
    denom = sum(abs(vector) .* abs(target));
    q = (numel ./ denom).^2;
end

