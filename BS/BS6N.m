function [b, bx, bxx, bxxx, bx4, bx5, Nor] = BS6N(x, s, k, id1, id2, h, H, ind)

    % Evaluate one compactly supported normalized B-spline and, optionally,
    % its first five derivatives at the points x. The interpolation builders
    % call this repeatedly for each local knot group.
    %
    % Outputs:
    %   b, bx, ... are derivative rows d_x^j B(x).
    %   Nor exposes the per-knot normalization constants used in the formula.

    % Follow the formula in App C.1 in Paper II to evaluate one B-spline function.
    %   B(x) =   sum_{0 <= j <= k} k (sj - x)^{k-1} * one_{sj-x >= 0} / d_j,      (1)
    %        = - sum_{0 <= j <=k } k (sj - x)^{k-1} * one_{sj-x < 0} / d_j,      (2)
    %    d_j = Prod_{ l \neq j} (s_j - s_l).  If sj - x =0, the term (sj-x)^{k-1} = 0
    % The identities come from:     sum_{0 <= j <= k} k * (sj - x)^{k-1}  / d_j = 0  (3)
    % which can be proved by comparing the coefficents of s^k on both sides for
    %     f(s) = sum_{0<=j<=k} f(sj) / d_j Prod_{j neq i} (s-sj),  f = s^l, l <= k-1
    % LHS, coefficient of s^k is 0;  RHS coe of s^k is sum f(sj) / dj * 1 = 0.
    % This identity is true for all l <= k-1, and implies (3)

    % k is the order, s are the supporting points; that is, b is piece-wise polynomial on each
    % interval (s_i,s_{i+1}),  x are the point to be evaluated at.
    % ind : indicate the number of derivatives to evaluate. default ind = 2

    % id1, id2 indicates location of the supporting points near 0 (id1) and in the far-field (id2).
    % It is used for choosing normalization constant to reduce condition number, h for near, and hh for far-field.
    % If id1 or id2 > 8 then the normalization factor independdent of localtion

    % The original verified MATLAB code also supports interval arithmetic.
    % Keep the same branches so non-double interval inputs remain usable.
    itl = ~isa(x, 'double') || ~isa(s, 'double'); % Check interval input

    if nargin == 7
        ind = 2;
    end

    if k ~= length(s) - 1
        error('size wrong !');
    end

    % First, sort the supporting points
    if itl
        s = sort_itl(s);
    else
        s = sort(s);
    end

    % Reshape s, so that if x is column, then s is row, id = 2;  or x is row, then s is column, id = 1.
    % Later, we sum over the s direction id.
    sx = size(x);
    if sx(2) == 1 % x column vector
        s = reshape(s, 1, length(s));
        dir_s = 2;
    else % x row vector
        s = reshape(s, length(s), 1);
        dir_s = 1;
    end

    % for x close to s(1), and to s(end), apply two different summation formulas (they are equivalent)
    % to reduce the round off error. See the top. For x large (lg), use (1); for x small use (2)
    % Spline support in [s(1), s(end)]
    isspt = (x > s(1)) & (x < s(end));
    islg = (x > s(1 + floor(k / 2))) & (x < s(end));
    issm = isspt - islg;

    if itl
        d = intval(zeros(size(s)));
    else
        d = zeros(size(s));
    end

    % d(i) = \prod_{j neq i} (s_i - s_j)
    for i = 1:k + 1
        v = s(i) - s;
        v(i) = 1;
        d(i) = prod(v);
    end

    % The sign expression chooses formula (1) or (2) above without explicitly
    % branching per point. That is important near knots, where the two
    % algebraically identical sums have different roundoff behavior.
    if itl
        sgn = (sign_itl(max(s - x, 0)) - issm) .* isspt;
    else
        sgn = (sign(max(s - x, 0)) - issm) .* isspt;
    end

    % Evaluate d_x^i B(x) using the top formula (1) or (2).
    % We further multiply the factor k * (k-1) * (k-2).. later
    Db = @(i) sum(sgn .* (s - x) .^ (k - 1 - i) ./ d * (-1) ^ i, dir_s);

    b = k * Db(0);

    % normalize Bspline following App C.1 s.t. middle point is of order 1 to get better condition number
    if id1 <= 8
        m = h;
    elseif id2 <= 8
        m = H / 100;
    else
        md = floor(k / 2) + 1;
        m = (s(md + 1) - s(md - 1)) / 2;
    end

    % Below, the factor k is from definition. Further * (k-1) * (k-2).. (k-i) from the derivatives.
    Nor = k * m ./ d;

    b = b * m;

    if ind ~= 1
        bx = Db(1) .* m * k * (k - 1);
    else
        bx = [];
    end

    if ind >= 3
        bxx = Db(2) .* m * k * (k - 1) * (k - 2);
    else
        bxx = [];
    end

    if ind >= 4
        bxxx = Db(3) .* m * k * (k - 1) * (k - 2) * (k - 3);
    else
        bxxx = [];
    end

    if ind >= 5
        bx4 = Db(4) .* m * k * (k - 1) * (k - 2) * (k - 3) * (k - 4);
        bx5 = Db(5) .* m * k * (k - 1) * (k - 2) * (k - 3) * (k - 4) * (k - 5);
    else
        bx4 = [];
        bx5 = [];
    end

end
