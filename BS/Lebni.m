function wBi = Lebni(Bi, Fun, xx)
    % compute the derivatives of  Bi * W using Leibniz rule.
    % Bi(k, : ) is the k-1 th derivative of Bi, Fun{k} is k-1 derivatives of W
    %
    % The interpolation matrices can include a coordinate weight W. For the
    % current optimizer W is usually constant, but keeping this helper makes the
    % spline assembly match the original verified MATLAB routines.

    m = size(Bi, 1);

    % Xg caches W and its derivatives at the evaluation points so the nested
    % loop below only combines arrays.
    Xg = zeros(m, length(xx));
    wBi = zeros(size(Bi));

    if isa(Bi, 'double') == 0 || isa(xx, 'double') == 0 % intval input
        Xg = intval(Xg);
        wBi = intval(wBi);
    end

    for i = 1:m
        Xg(i, :) = Fun{i}(xx);
    end

    for ii = 1:m

        for j = 1:ii
            % Xg(ii+1-j, ) for d_x^{ii-j} W, Bi(j,:) is d_x^{j-1} Bi; wBi(ii,:) for d_x^{ii-1} (W * Bi)
            % Matlab index starts from 1
            wBi(ii, :) = wBi(ii, :) + Bi(j, :) .* Xg(ii + 1 - j, :) * nchoosek(ii - 1, j - 1);
        end

    end

end
