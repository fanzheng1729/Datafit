function [knot_index, Spline_index, derivs, t] = prealloc(N, n)
    % Shared sparse-assembly scratch space for BS6_interp and BS6_interp2.
    % N is one plus the highest requested derivative count; n is a conservative
    % bound on the number of nonzero point/basis pairs to record.
    knot_index = zeros(n, 1);
    Spline_index = knot_index;
    % derivs(row, j) stores the j-1 derivative value for one nonzero entry.
    derivs = zeros(n, N - 1);
    % t is the active row count in the preallocated arrays.
    t = 0;
end
