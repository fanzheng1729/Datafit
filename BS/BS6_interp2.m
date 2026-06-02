function BS = BS6_interp2(x1, x2, z1, z2, Fun, r1, extrap, trim, ind)

    % Even-x version of BS6_interp.
    %
    % Main changes from the odd-x version:
    %   1. X-direction has one extra spline from xx1(1:N).
    %   2. X-direction loop uses i = 0:N1-1.
    %   3. The i = 0 spline is self-reflected, so use Bi = 2 * Bi.
    %   4. For i > 0 near x = 0, use Bi + reflected Bi instead of Bi - reflected Bi.
    %   5. X-direction has N1 basis functions instead of N1 - 1.
    %
    % Y-direction is unchanged.

    if length(find(z1 == 0)) > 1 || length(find(z2 == 0)) > 1
        error('Too many zeros!');
    end

    h1 = x1(2) - x1(1);
    h2 = x2(2) - x2(1);
    H1 = x1(end) - x1(end - 1);
    H2 = x2(end) - x2(end - 1);

    N = 7;
    coe = BScoe(N);

    N1 = length(x1);
    N2 = length(x2);

    x1 = reshape(x1, 1, N1);
    x2 = reshape(x2, 1, N2);

    if ~isstruct(r1)
        ad1 = Meshext(x1, N - 2, "exp", r1);
        ad2 = Meshext(x2, N - 2, "exp", r1);
    else
        ad1 = reshape(r1.ex{1}(N1 + 1:N1 + N - 2), 1, []);
        ad2 = reshape(r1.ex{2}(N2 + 1:N2 + N - 2), 1, []);
    end

    xx1 = [-x1(4), -x1(3), -x1(2), x1, ad1];
    xx2 = [-x2(4), -x2(3), -x2(2), x2, ad2];

    BS = cell(N - 1, 2);

    %% X-direction: bulk splines, including the extra even central spline

    [x_index, B_index, derivs, t] = prealloc(ind + 1, (N - 2) * (N1 + 1) + extrap * N);

    for i = 0:N1 - 1
        x_knots = xx1(i + 1:i + N);

        % z1 = 0 is treated specially below.
        z_indices = find((z1 - x_knots(1)) .* (z1 - x_knots(end)) < 0 & z1 ~= 0);
        len_z = length(z_indices);

        if len_z > 0

            z_coords = z1(z_indices);

            if ~extrap
                pos = Inf;
            else
                pos = N1 - i - 1;
            end

            Bi = zeros(ind, len_z);
            [Bi(min(1, ind), 1:end * (ind >= 1)), ...
                Bi(min(2, ind), 1:end * (ind >= 2)), ...
                Bi(min(3, ind), 1:end * (ind >= 3)), ...
                Bi(min(4, ind), 1:end * (ind >= 4)), ...
                Bi(min(5, ind), 1:end * (ind >= 5)), ...
                Bi(min(6, ind), 1:end * (ind >= 6))] = ...
                BS6N(z_coords, x_knots, N - 1, i - 1, pos, h1, H1, ind);

            if i == 0
                % Extra central spline from xx1(1:N).
                % Its reflection is itself, so the even extension is 2 * Bi.
                Bi = 2 * Bi;

            elseif i <= floor(N / 2) - 1
                % Near-origin reflected spline.
                % Even extension: B_i(x) + B_i(-x).
                rBi = zeros(ind, len_z);

                [rBi(min(1, ind), 1:end * (ind >= 1)), ...
                    rBi(min(2, ind), 1:end * (ind >= 2)), ...
                    rBi(min(3, ind), 1:end * (ind >= 3)), ...
                    rBi(min(4, ind), 1:end * (ind >= 4)), ...
                    rBi(min(5, ind), 1:end * (ind >= 5)), ...
                    rBi(min(6, ind), 1:end * (ind >= 6))] = ...
                    BS6N(z_coords, -x_knots, N - 1, i - 1, pos, h1, H1, ind);

                Bi = Bi + rBi;
            end

            wBi = Lebni(Bi, Fun(1, :), z_coords);

            x_index(t + 1:t + len_z) = z_indices;
            B_index(t + 1:t + len_z) = i + 1;
            derivs(t + 1:t + len_z, :) = wBi';
            t = t + len_z;
        end
    end

    %% X-direction: far-field extrapolation

    if extrap
        z_indices = find((z1 - x1(N1 - 4)) .* (z1 - xx1(end)) < 0);
        len_z = length(z_indices);

        if len_z > 0

            z_coords = z1(z_indices);

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

                A = A1 * coe(2, N1 - i) + A2 * coe(1, N1 - i);
                wA = Lebni(A, Fun(1, :), z_coords);

                % Shifted by one because x-direction now has the extra i = 0 basis.
                x_index(t + 1:t + len_z) = z_indices;
                B_index(t + 1:t + len_z) = i + 1;
                derivs(t + 1:t + len_z, :) = wA';
                t = t + len_z;
            end
        end
    end

    %% X-direction: origin for even symmetry

    if z1(1) == 0

        for i = 0:floor(N / 2) - 1
            knots = xx1(i + 1:i + N);

            A1 = zeros(ind, 1);

            % At x = 0, only even derivative orders survive:
            % row 1 = 0th derivative, row 3 = 2nd derivative, row 5 = 4th derivative.
            [A1(min(1, ind), 1:end * (ind >= 1)), ...
                ~, ...
                A1(min(3, ind), 1:end * (ind >= 3)), ...
                ~, ...
                A1(min(5, ind), 1:end * (ind >= 5)), ...
                ~] = ...
                BS6N(0, knots, N - 1, i - 1, N1 - 1, h1, H1, ind);

            if i == 0
                % Self-reflected central basis.
                wA = Lebni(2 * A1, Fun(1, :), 0);
            else
                A2 = zeros(ind, 1);

                [A2(min(1, ind), 1:end * (ind >= 1)), ...
                    ~, ...
                    A2(min(3, ind), 1:end * (ind >= 3)), ...
                    ~, ...
                    A2(min(5, ind), 1:end * (ind >= 5)), ...
                    ~] = ...
                    BS6N(0, -knots, N - 1, i - 1, N1 - 1, h1, H1, ind);

                wA1 = Lebni(A1, Fun(1, :), 0);
                wA2 = Lebni(A2, Fun(1, :), 0);

                wA = wA1 + wA2;
            end

            t = t + 1;
            x_index(t) = 1;
            B_index(t) = i + 1;
            derivs(t, :) = wA';
        end
    end

    %% Assemble x-direction matrices

    if trim == 0
        nBS = N1;
    else
        nBS = max(B_index(1:t));
    end

    for ord = 1:ind
        BS{ord, 1} = sparse(x_index(1:t), B_index(1:t), derivs(1:t, ord), length(z1), nBS);
    end

    clear x_index B_index derivs

    %% Y-direction: unchanged from original BS6_interp

    [x_index2, B_index2, derivs2, t] = prealloc(ind + 1, (N - 2) * N2 + (extrap + 1) * N);

    for j = 0:N2 - 1
        x_knots = xx2(j + 1:j + N);
        z_indices = find((z2 - x_knots(1)) .* (z2 - x_knots(end)) < 0);
        len_z = length(z_indices);

        if len_z > 0

            z_coords = z2(z_indices);

            if ~extrap
                pos = Inf;
            else
                pos = N2 - j - 1;
            end

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

    %% Y-direction: near-boundary extrapolation

    ls01 = -xx2(2:8);
    ls02 = -xx2(3:9);

    z_indices = find((z2 - min([ls01, ls02])) .* (z2 - max([ls01, ls02])) < 0);
    z_coords = z2(z_indices);
    len_z = length(z_indices);

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
        wA = Lebni(A, Fun(2, :), z_coords);

        x_index2(t + 1:t + len_z) = z_indices;
        B_index2(t + 1:t + len_z) = j + 1;
        derivs2(t + 1:t + len_z, :) = wA';
        t = t + len_z;
    end

    %% Y-direction: far-field extrapolation

    if extrap
        z_indices = find((z2 - x2(N2 - 4)) .* (z2 - xx2(end)) < 0);
        len_z = length(z_indices);

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

                A = A1 * coe(2, N2 - j) + A2 * coe(1, N2 - j);
                wA = Lebni(A, Fun(2, :), z_coords);

                x_index2(t + 1:t + len_z) = z_indices;
                B_index2(t + 1:t + len_z) = j + 1;
                derivs2(t + 1:t + len_z, :) = wA';
                t = t + len_z;
            end
        end
    end

    %% Assemble y-direction matrices

    if trim == 0
        nBS = N2;
    else
        nBS = max(B_index2(1:t));
    end

    for ii = 1:ind
        BS{ii, 2} = sparse(x_index2(1:t), B_index2(1:t), derivs2(1:t, ii), length(z2), nBS);
    end

end