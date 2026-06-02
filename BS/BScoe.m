function coemat = BScoe(N, rows)

    % Compute the coefficients of B-splines used for extrapolation.

    % BScoe(7) =
    % 28 -112 210 -224 140 -48 7
    %  7 -21  35  -35  21  -7  1
    
    % BScoe(9) =
    % 165 -990 2772 -4620 4950 -3465 1540 -396 45
    %  45 -240 630  -1008 1050 -720  315  -80  9
    %  9  -36  84   -126  126  -84   36   -9   1
    
    if nargin == 1
        rows = floor(N / 2) - 1;
    end
    [I, J] = ndgrid(rows - 1:-1:0, 1:N);
    coemat = arrayfun(@(i, j) ...
        (-1)^(j - 1) * nchoosek(N + i, i + j) * nchoosek(i + j - 1, i), ...
        I, J);

end
