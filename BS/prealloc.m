function [knot_index, Spline_index, derivs, t] = prealloc(N, n)
    knot_index = zeros(n, 1);
    Spline_index = knot_index;
    derivs = zeros(n, N - 1);
    t = 0;
end