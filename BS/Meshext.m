function ad = Meshext(x, k, flag, r)
    % Extend the mesh x beyond x(end) for k more grids using two ways. 
    % return ad the extension. It has same form, i.e. row or column vectors as x

    if flag == "uni"
        % uniform extension
        h = x(end) - x(end - 1);
        ad = ones(1, k) * x(end) + (1:k) * h;
    elseif flag == "exp"
        % exponential extend
        %r = x(end) / x(end - 1);
        ad = x(end) .* r.^(1:k);

    else
        error('Not match');
    end

    % If x is column vector, ad is so
    if size(x, 1) > 1
        ad = ad';
    end

end
