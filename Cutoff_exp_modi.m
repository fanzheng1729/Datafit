function g = Cutoff_exp_modi(f, x, a, k)
    % f = 1 / (1 + exp( 1/ x + 1 / (x-a)) ),
    % near 0,  f(x) can be Nan due to exp(1/ x + 1/ (x-a)) = Inf! Handle the special case
    % k is the number of derivatives, k >= 0; modify it using equivalent formula

    g = zeros(size(x));

    if isa(x, 'double') == 0 % intval input
        g = intval(g);
    end

    id = find(x > a / 2 & x < a);
    id2 = find(x > 0 & x <= a / 2);
    g(x <= 0) = 0;

    if k >= 1
        g(x >= a) = 0;
    else
        g(x >= a) = 1;
    end

    g(id) = f(x(id));
    % f(x) + f(a-x) = 1, d_k f(x) = - (-1)^k * d_k f(a-x)
    if k >= 1
        g(id2) = (-1) ^ (k + 1) * f(a - x(id2));
    else
        g(id2) = 1 - f(a - x(id2));
    end

end
