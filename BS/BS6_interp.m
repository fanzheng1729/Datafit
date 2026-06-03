function BS = BS6_interp(x1, x2, z1, z2, Fun, r1, extrap, trim, ind)

    % Build sparse one-dimensional BS6 matrices in both coordinate directions.
    % The returned cell has BS{d,1} for x-direction d-1 derivatives and
    % BS{d,2} for y-direction d-1 derivatives.
    %
    % This is the odd-x version used for omega, zeta, and u1.

    % Follow App C.1 in paper II to build the Bspline matrix in i-direction with supporting points on xi,
    %  sum a_ij B_{1,i}(z1) * B_{2,j}(z2) * F2(z2)  = B(z1, i) * A_ij * B(z2, j)' * F2(z2)
    % The spline in x-direction is odd at 0. in y-direction, use extrapolation near the boundary.
    % F2 = Fun{2} is the weight in Y-direction. X-direction weight is always 1
    % the Spline is only C^^{4, 1}; d_x^5 or d_y^5 is NOT continuous across the supporting points x1, x2

    % Options:
    % extrap = 1, for stream function, also use extrapolation in the far-field. See App C.1 in Paper II
    % extrap = 0, no extrapolation. Used for w and theta / x1

    % trim = 1, do not build a full BS matrix. this cases zi is near 0, where the Bsplines in the far-field
    % are 0 on zi and do not contribute to zi.
    % ord = 0, only compute the matrix evaluating the 0-order value (not derivatives)

    % 3 steps (1) Assemble Bspline matrix in x-direction from the bulk spline
    %         (2) Handle the spline due to extrapolation and the value at 0
    %         (3) Do for y-direction similarly. Bulk, then extrapolation near boundary and far-field

    if length(find(z1 == 0)) > 1 || length(find(z2 == 0)) > 1
        error('Too many zeros!');
    end

    % hi0, hhi used to normalize the spline
    h1 = x1(2) - x1(1);
    h2 = x2(2) - x2(1);
    H1 = x1(end) - x1(end - 1);
    H2 = x2(end) - x2(end - 1);

    % coefficients for 7-th order extrapolation. We always use this set of parameters
    % and the SAME combination of 7-th order extrapolated Bspline basis for representation.
    % Thus, NOT neccessary to verify these formulas.
    N = 7;
    coe = BScoe(N);

    N1 = length(x1);
    N2 = length(x2);

    x1 = reshape(x1, 1, N1);
    x2 = reshape(x2, 1, N2);

    % If r1 is a structure, r1.ex contains the extension of xi, which contains all the supporting points
    % for all splines; otherwise, get the extra supporting points by extending x1
    if ~isstruct(r1)
        ad1 = Meshext(x1, N - 2, "exp", r1);
        ad2 = Meshext(x2, N - 2, "exp", r1);
    else
        ad1 = reshape(r1.ex{1}(N1 + 1:N1 + N - 2), 1, []);
        ad2 = reshape(r1.ex{2}(N2 + 1:N2 + N - 2), 1, []);
    end

    xx1 = [-x1(4), -x1(3), -x1(2), x1, ad1];
    xx2 = [-x2(4), -x2(3), -x2(2), x2, ad2];

    % Store derivative matrices separately by derivative order and coordinate
    % direction. The interpolation routines only fill the requested ind rows.
    BS = cell(N - 1, 2);

    [x_index, B_index, derivs, t] = prealloc(ind + 1, (N - 2) * N1 + extrap * N);
    % Step 1: X-direction: Contributions from bulk Bspline
    for i = 1:N1 - 1
        x_knots = xx1(i + 1:i + N);
        % 0 is treated specially below, find z1 in the support of the spline [ls(1), ls(end)]
        z_indices = find((z1 - x_knots(1)) .* (z1 - x_knots(end)) < 0 & z1 ~= 0);
        len_z = length(z_indices);

        % Only need to do the computation if J is not empty
        if len_z > 0

            z_coords = z1(z_indices);
            % pos is used to select one of two types of normalization constant of the Spline in the far-field
            % extrap = 0: Case without far-field extrapolation, set pos >= 15. Then it only select the standard
            % type of normalization constant of Bspline in the far-field. See C_i at end App. C.1 in Paper II
            if ~extrap
                pos = Inf;
            else
                pos = N1 - i - 1;
            end

            % Bi rows are derivative orders, columns are the selected z points.
            Bi = zeros(ind, len_z);
            [Bi(min(1, ind), 1:end * (ind >= 1)), ...
                Bi(min(2, ind), 1:end * (ind >= 2)), ...
                Bi(min(3, ind), 1:end * (ind >= 3)), ...
                Bi(min(4, ind), 1:end * (ind >= 4)), ...
                Bi(min(5, ind), 1:end * (ind >= 5)), ...
                Bi(min(6, ind), 1:end * (ind >= 6))] = ...
                BS6N(z_coords, x_knots, N - 1, i - 1, pos, h1, H1, ind);

            if i <= floor(N / 2) - 1
                rBi = zeros(ind, len_z);
                % In x-direction, near 0, the spline is odd, first 2 splines: B(z, s) - B(z, -s). See App C.1
                % We have B(z; -s) = B(-z; s).
                [rBi(min(1, ind), 1:end * (ind >= 1)), ...
                    rBi(min(2, ind), 1:end * (ind >= 2)), ...
                    rBi(min(3, ind), 1:end * (ind >= 3)), ...
                    rBi(min(4, ind), 1:end * (ind >= 4)), ...
                    rBi(min(5, ind), 1:end * (ind >= 5)), ...
                    rBi(min(6, ind), 1:end * (ind >= 6))] = ...
                    BS6N(z_coords, -x_knots, N - 1, i - 1, pos, h1, H1, ind);
                % B_i(x) = B_i(x) - B_i(-x) to make it odd
                Bi = Bi - rBi;
            end

            % Lebnize rule, compute  d_x^i ( Bi0 * F1(x)), F1 is x direction wg with d_x^i F1 = Fun(1, i+1)
            wBi = Lebni(Bi, Fun(1, :), z_coords);

            % Accumulate triplets for one sparse matrix entry per
            % evaluation-point/basis pair.
            x_index(t + 1:t + len_z) = z_indices;
            B_index(t + 1:t + len_z) = i;
            derivs(t + 1:t + len_z, :) = wBi';
            t = t + len_z;
        end

    end

    % Step 2: contribution from two far-field extrapolation basis with supporting points at ls23, ls24
    % and then value at 0.
    if extrap
        % extrapolaton, find all the grids in the support
        z_indices = find((z1 - x1(N1 - 4)) .* (z1 - xx1(end)) < 0);
        len_z = length(z_indices);

        if len_z > 0

            z_coords = z1(z_indices);
            % Supporting points for the two extrapolated Spline in the far-field
            last2 = xx1(end - N:end - 1);
            last = xx1(end - N + 1:end);
            A1 = zeros(ind, len_z);
            A2 = A1;
            for i = N1 - N:N1 - 1
                [A1(min(1, ind), 1:end * (ind >= 1)), ...
                    A1(min(2, ind), 1:end * (ind >= 2)), ...
                    A1(min(3, ind), 1:end * (ind >= 3)), ...
                    A1(min(4, ind), 1:end * (ind >= 4)), ...
                    A1(min(5, ind), 1:end * (ind >= 5)), ...
                    A1(min(6, ind), 1:end * (ind >= 6))] = ...
                    BS6N(z_coords, last2, N - 1, N1, 0, h1, H1, ind);
                [A2(min(1, ind), 1:end * (ind >= 1)), ...
                    A2(min(2, ind), 1:end * (ind >= 2)), ...
                    A2(min(3, ind), 1:end * (ind >= 3)), ...
                    A2(min(4, ind), 1:end * (ind >= 4)), ...
                    A2(min(5, ind), 1:end * (ind >= 5)), ...
                    A2(min(6, ind), 1:end * (ind >= 6))] = ...
                    BS6N(z_coords, last, N - 1, N1, 0, h1, H1, ind);
                % Bext = coe(2, N:1) * 2nd last B + coe(1, N:1) * last B
                A = A1 * coe(2, N1 - i) + A2 * coe(1, N1 - i);
                % Leibniz rule, compute k-th derivatives of Bi * Xg
                wA = Lebni(A, Fun(1, :), z_coords);

                x_index(t + 1:t + len_z) = z_indices;
                B_index(t + 1:t + len_z) = i;
                derivs(t + 1:t + len_z, :) = wA';
                t = t + len_z;
            end

        end

    end

    %%
    % nBS = #B-splines used
    if trim == 0
        nBS = N1 - 1;
    else
        % do not built a long sparse matrix. See top for meaning of this option
        nBS = max(B_index);
    end
    % Assemble 0, 2, 4 derivatives matrix.
    for ord = 1:2:ind
        BS{ord, 1} = sparse(x_index(1:t), B_index(1:t), derivs(1:t, ord), length(z1), nBS);
    end
    %%
    % origin
    if z1(1) == 0
        % Handle value at 0 specially to impose the odd conditions.
        A1 = zeros(ind, 1);
        A2 = A1;
        for i = 1:floor(N / 2) - 1
            % 2nd and 3rd knot groups
            knots = xx1(i + 1:i + N);
            % 1,3,5 rows cancel
            [~, ...
                A1(min(2, ind), 1:end * (ind >= 2)), ...
                ~, ...
                A1(min(4, ind), 1:end * (ind >= 4)), ...
                ~, ...
                A1(min(6, ind), 1:end * (ind >= 6))] = ...
                BS6N(0, knots, N - 1, 1, N1 - 1, h1, H1, ind);
            [~, ...
                A2(min(2, ind), 1:end * (ind >= 2)), ...
                ~, ...
                A2(min(4, ind), 1:end * (ind >= 4)), ...
                ~, ...
                A2(min(6, ind), 1:end * (ind >= 6))] = ...
                BS6N(0, -knots, N - 1, 1, N1 - 1, h1, H1, ind);
            % Use Leibniz rule to evaluate d_x^j( Ai * F1)
            wA1 = Lebni(A1, Fun(1, :), 0);
            wA2 = Lebni(A2, Fun(1, :), 0);
            % (1, i) = 2B_i'(0) (not exactly due to floating point error
            % and at 5th derivative
            t = t + 1;
            x_index(t) = 1;
            B_index(t) = i;
            derivs(t, :) = wA1' - wA2';
        end

    end
    % Assemble 1, 3, 5 derivatives matrix.
    for ord = 2:2:ind
        BS{ord, 1} = sparse(x_index(1:t), B_index(1:t), derivs(1:t, ord), length(z1), nBS);
    end

    clear x_index B_index derivs
    %%
    % Step 3: Get the spline matrix in Y-direction. Bulk + two extrapolations. Similar to the X-direction
    % ls01, ls02 supporting points for extra spline near boundary; ls23, ls24 for far-field
    % ls0j for B_{-j}(y), center at -j * h, xx2(i) = (i-4) * h
    [x_index2, B_index2, derivs2, t] = prealloc(ind + 1, (N - 2) * N2 + (extrap + 1) * N);
    for j = 0:N2 - 1
        % from the 1nd to last 3rd knot groups in xx2
        x_knots = xx2(j + 1:j + N);
        z_indices = find((z2 - x_knots(1)) .* (z2 - x_knots(end)) < 0);
        len_z = length(z_indices);

        if len_z > 0

            z_coords = z2(z_indices);
            % pos is used to select one of two types of normalization constant of the Spline in the far-field
            % same reason as the case in X-direction. See above
            if ~extrap
                pos = Inf;
            else
                pos = N2 - j - 1;
            end

            % Unlike x, y has no odd reflection at the origin in this builder.
            Bj = zeros(ind, len_z);
            [Bj(min(1, ind), 1:end * (ind >= 1)), ...
                Bj(min(2, ind), 1:end * (ind >= 2)), ...
                Bj(min(3, ind), 1:end * (ind >= 3)), ...
                Bj(min(4, ind), 1:end * (ind >= 4)), ...
                Bj(min(5, ind), 1:end * (ind >= 5)), ...
                Bj(min(6, ind), 1:end * (ind >= 6))] = ...
                BS6N(z_coords, x_knots, N - 1, j - 1, pos, h2, H2, ind);

            wBj = Lebni(Bj, Fun(2, :), z_coords);

            x_index2(t + 1:t + len_z) = z_indices;
            B_index2(t + 1:t + len_z) = j + 1;
            derivs2(t + 1:t + len_z, :) = wBj';
            t = t + len_z;
        end

    end

    %%
    % contribution from B_-1 and B_-2
    % two knot groups to the left of the 1st one
    ls01 = -xx2(2:8);
    ls02 = -xx2(3:9);
    z_indices = find((z2 - min([ls01, ls02])) .* (z2 - max([ls01, ls02])) < 0);
    z_coords = z2(z_indices);
    len_z = length(z_indices);

    % Extrapolation near the boundary. Contribution from spline with supporting points at ls01, ls02
    for j = 0:N - 1

        A1 = zeros(ind, len_z);
        A2 = A1;
        [A1(min(1, ind), 1:end * (ind >= 1)), ...
            A1(min(2, ind), 1:end * (ind >= 2)), ...
            A1(min(3, ind), 1:end * (ind >= 3)), ...
            A1(min(4, ind), 1:end * (ind >= 4)), ...
            A1(min(5, ind), 1:end * (ind >= 5)), ...
            A1(min(6, ind), 1:end * (ind >= 6))] = ...
            BS6N(z_coords, ls01, N - 1, 0, N2, h2, H2, ind);
        [A2(min(1, ind), 1:end * (ind >= 1)), ...
            A2(min(2, ind), 1:end * (ind >= 2)), ...
            A2(min(3, ind), 1:end * (ind >= 3)), ...
            A2(min(4, ind), 1:end * (ind >= 4)), ...
            A2(min(5, ind), 1:end * (ind >= 5)), ...
            A2(min(6, ind), 1:end * (ind >= 6))] = ...
            BS6N(z_coords, ls02, N - 1, 0, N2, h2, H2, ind);

        A = A1 * coe(2, j + 1) + A2 * coe(1, j + 1);
        % Fun(2, :), weight in y direction
        wA = Lebni(A, Fun(2, :), z_coords);

        x_index2(t + 1:t + len_z) = z_indices;
        B_index2(t + 1:t + len_z) = j + 1;
        derivs2(t + 1:t + len_z, :) = wA';
        t = t + len_z;
    end

    if extrap
        z_indices = find((z2 - x2(N2 - 4)) .* (z2 - xx2(end)) < 0);
        len_z = length(z_indices);
        % contribution from two far-field extrapolation basis with supporting points at ls23, ls24

        if len_z > 0

            z_coords = z2(z_indices);
            last2 = xx2(end - N:end - 1);
            last = xx2(end - N + 1:end);
            A1 = zeros(ind, len_z);
            A2 = A1;
            for j = N2 - N:N2 - 1
                [A1(min(1, ind), 1:end * (ind >= 1)), ...
                    A1(min(2, ind), 1:end * (ind >= 2)), ...
                    A1(min(3, ind), 1:end * (ind >= 3)), ...
                    A1(min(4, ind), 1:end * (ind >= 4)), ...
                    A1(min(5, ind), 1:end * (ind >= 5)), ...
                    A1(min(6, ind), 1:end * (ind >= 6))] = ...
                    BS6N(z_coords, last2, N - 1, N2, 0, h2, H2, ind);
                [A2(min(1, ind), 1:end * (ind >= 1)), ...
                    A2(min(2, ind), 1:end * (ind >= 2)), ...
                    A2(min(3, ind), 1:end * (ind >= 3)), ...
                    A2(min(4, ind), 1:end * (ind >= 4)), ...
                    A2(min(5, ind), 1:end * (ind >= 5)), ...
                    A2(min(6, ind), 1:end * (ind >= 6))] = ...
                    BS6N(z_coords, last, N - 1, N2, 0, h2, H2, ind);
                % B_j += coe(2, N:1) * 2nd last B + coe(1, N:1) * last B
                A = A1 * coe(2, N2 -j) + A2 * coe(1, N2 - j);
                % Leibniz rule, compute k-th derivatives of Bi * Xg
                wA = Lebni(A, Fun(2, :), z_coords);

                x_index2(t + 1:t + len_z) = z_indices;
                B_index2(t + 1:t + len_z) = j + 1;
                derivs2(t + 1:t + len_z, :) = wA';
                t = t + len_z;
            end

        end

    end

    % nBS = #B-splines used
    if trim == 0
        nBS = N2;
    else
        % trim = 1. do not built a long sparse matrix. See top for this option.
        nBS = max(B_index2);
    end

    % Y has no origin parity split, so all derivative orders can be assembled
    % in one pass.
    for ii = 1:ind
        BS{ii, 2} = sparse(x_index2(1:t), B_index2(1:t), derivs2(1:t, ii), length(z2), nBS);
    end

end
