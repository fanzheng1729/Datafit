% Fit and residual checks for the Boussinesq profile in Analysis.pdf and Numerics.pdf.
%
% Paper notation:
%   w     in data.mat is omega, the vorticity profile.
%   v     in data.mat is zeta = theta / x1. Thus theta_x = zeta + x1*zeta_x1.
%   u1,u2 are the velocity components from the stream function.
%   cl,cw are the dynamic-rescaling rates c_l and c_omega from Analysis (2.10)-(2.11).
%
% The residual checks below are the steady versions of Analysis (2.10) after
% replacing theta by x1*zeta:
%   Fomega = -(cl*x+u).grad omega + cw*omega + theta_x
%   Fzeta  = -(cl*x+u).grad zeta + (2*cw - u1/x1)*zeta
%
% The finite-dimensional representation follows Numerics Section 7 and Appendix C:
% sixth-order tensor-product B-splines, odd/even symmetry at x1=0, far-field
% extrapolation, and a semi-analytic far-field stream-function contribution.
%
% Forcing in run.m
% warning off
% [~, ~, Vel, ~, wx1, wx2, vx1, vx2] = F_df(v, w, rec, [], []);
% warning on
addpath("BS\")
load data.mat
omega = w;
omega_x1 = wx1;
omega_x2 = wx2;
zeta = v;
zeta_x1 = vx1;
zeta_x2 = vx2;

% Fit omega and zeta using SVD + B-splines.
[omegafit, omegafitx1, omegafitx2] = fitprofilescaled(omega, x1, x2, 1e-10, 1e-10);
[zetafit, zetafitx1, zetafitx2] = fitprofilescaled(zeta, x1, x2, 1e-10, 1e-10);
[rms(omega-omegafit,"all") rms(omega_x1-omegafitx1,"all") rms(omega_x2-omegafitx2,"all")]
[rms(zeta-zetafit,"all") rms(zeta_x1-zetafitx1,"all") rms(zeta_x2-zetafitx2,"all")]
% [omegafit, omegafitx1, omegafitx2] = SVDfit(omega, x1, x2, 1e-10, 1e-10);
% [zetafit, zetafitx1, zetafitx2] = SVDfit(zeta, x1, x2, 1e-10, 1e-10);
% [rms(omega-omegafit,"all") rms(omega_x1-omegafitx1,"all") rms(omega_x2-omegafitx2,"all")]
% [rms(zeta-zetafit,"all") rms(zeta_x1-zetafitx1,"all") rms(zeta_x2-zetafitx2,"all")]
psi = Vel.psi;
psi0f = Vel.psi0f;
u1 = Vel.u1;
u2 = Vel.u2;
u10f = Vel.u10f;
u20f = Vel.u20f;
rat = rec(7);
[u10ffit, u10fx1, u10fx2] = fitu1damped(u10f, x1, x2, 1e-11, 1e-11, 1/2);
Psi1 = Deri_Psi1(length(gx1), length(gx2), p_ag_coe, BS1d_large, 2, XYcoef(gx1, gx2, alpha_b, Chi20, AG));
Psi1 = Cell_2double(Psi1);
n1 = length(x1);
n2 = length(x2);
u1fit = u10ffit - rat * Psi1(1:n1, 1:n2, 1, 2);
u1x1  = u10fx1  - rat * Psi1(1:n1, 1:n2, 2, 2);
u1x2  = u10fx2  - rat * Psi1(1:n1, 1:n2, 1, 3);
[u20ffit, u20fx1, u20fx2] = fitu2damped(u20f, x1, x2, 1e-11, 1e-11, 1/2);
u2fit = u20ffit + rat * Psi1(1:n1, 1:n2, 2, 1);
u2x1  = u20fx1  + rat * Psi1(1:n1, 1:n2, 3, 1);
u2x2  = u20fx2  + rat * Psi1(1:n1, 1:n2, 2, 2);
rms(u1-u1fit,"all")/rms(u1,"all")
rms(u2-u2fit,"all")/rms(u2,"all")
rms(u1x1+u2x2,"all")
% curl(u) = u1_x2 - u2_x1 should reproduce omega.
rms(u1x2-u2x1-omegafit,"all")
cl = 4 * zeta_x1(1, 1) / omega_x1(1, 1);
cw = Vel.u1dx1(1, 1) + cl / 2;
max(abs(Fomega(cl, cw, x1, x2, omega, zeta, omega_x1, omega_x2, u1, u2, zeta_x1)), [], "all")
max(abs(Fomega(cl, cw, x1, x2, omegafit, zetafit, omegafitx1, omegafitx2, u1fit, u2fit, zetafitx1)), [], "all")
rms(abs(Fomega(cl, cw, x1, x2, omega, zeta, omega_x1, omega_x2, u1, u2, zeta_x1)), "all")
rms(abs(Fomega(cl, cw, x1, x2, omegafit, zetafit, omegafitx1, omegafitx2, u1fit, u2fit, zetafitx1)), "all")
max(abs(Fzeta(cl, cw, x1, x2, zeta, zeta_x1, zeta_x2, u1, u2)), [], "all")
max(abs(Fzeta(cl, cw, x1, x2, zetafit, zetafitx1, zetafitx2, u1fit, u2fit)), [], "all")
rms(abs(Fzeta(cl, cw, x1, x2, zeta, zeta_x1, zeta_x2, u1, u2)), "all")
rms(abs(Fzeta(cl, cw, x1, x2, zetafit, zetafitx1, zetafitx2, u1fit, u2fit)), "all")

function res = Fomega(cl, cw, x1, x2, omega, zeta, omega_x1, omega_x2, u1, u2, zeta_x1)
    theta_x = zeta + x1 .* zeta_x1;
    res =- (cl * x1 + u1) .* omega_x1 - (cl * x2' + u2) .* omega_x2 + cw * omega + theta_x;
end
function res = Fzeta(cl, cw, x1, x2, zeta, zeta_x1, zeta_x2, u1, u2)
    u1dx1 = zeros(size(u1));
    u1dx1(2:end, :) = u1(2:end, :) ./ x1(2:end);
    res =- (cl * x1 + u1) .* zeta_x1 - (cl * x2' + u2) .* zeta_x2 + (2 * cw - u1dx1) .* zeta;
end

function [Afit, Afitx1, Afitx2] = fitprofilescaled(A, x1, x2, epsSVD, epsfit)
    factor = sqrt(1 + x1.^2) ./ sqrt(1 + x1.^2 + x2'.^2);
    Ascaled = A ./ factor;
    [Ascaledfit, Ascaledfitx1, Ascaledfitx2] = SVDfit(Ascaled, x1, x2, epsSVD, epsfit);
    Afit = Ascaledfit .* factor;
    Afitx1 = factor .* (Ascaledfitx1 + Ascaledfit .* x1 .* x2'.^2 ./ (1 + x1.^2) ./ (1 + x1.^2 + x2'.^2));
    Afitx2 = factor .* (Ascaledfitx2 - Ascaledfit .* x2' ./ (1 + x1.^2 + x2'.^2));
end

function [u1fit, u1x1, u1x2] = fitu1damped(u1, x1, x2, epsSVD, epsfit, power)
    if nargin < 6, power = 1; end
    factor0= 1 + x1.^2;
    factor = factor0 .^ (power/2);
    damped = u1 ./ factor;
    [dampedfit, dampedfitx1, dampedfitx2] = SVDfit(damped, x1, x2, epsSVD, epsfit);
    dampedfit(1, :) = 0;
    dampedfitx2(1, :) = 0;
    u1fit = dampedfit .* factor;
    u1x2  = dampedfitx2 .* factor;
    u1x1  = dampedfitx1 .* factor + power * x1 .* factor .* dampedfit ./ factor0;
end
function [u2fit, u2x1, u2x2] = fitu2damped(u2, x1, x2, epsSVD, epsfit, power)
    if nargin < 6, power = 1; end
    factor0= 1 + x2'.^2;
    factor = factor0 .^ (power/2);
    damped = u2 ./ factor;
    [dampedfit, dampedfitx1, dampedfitx2] = SVDfit(damped, x1, x2, epsSVD, epsfit, 0);
    dampedfit(:, 1) = 0;
    dampedfitx1(:, 1) = 0;
    u2fit = dampedfit .* factor;
    u2x1  = dampedfitx1 .* factor;
    u2x2  = dampedfitx2 .* factor + power * x2' .* factor .* dampedfit ./ factor0;
end

function [Afit, Afitx1, Afitx2, Afitx1x1, Afitx2x2] = SVDfit(A, x1, x2, epsSVD, epsfit, parity)
    odd = 1;
    if nargin < 6, parity = odd; end 
    [U, S, V] = svd(A);
    N = SVDorder(U, S, V, epsSVD);
    [U, S, V] = SVDtrunc(U, S, V, N);
    if parity == odd, U(1, :) = 0; end
    Ufit = U;
    Vfit = V;
    dUfit = zeros(size(U));
    dVfit = zeros(size(V));
    if nargout > 3
        ddUfit = zeros(size(U));
        ddVfit = zeros(size(V));
    end
    Ustep= zeros(size(U(1, :)));
    Vstep= Ustep;
    mat  = BS6mat(x1, x2, 1, parity);
    if parity == odd
        Ucoef = mat{1,1}(2:end, :) \ U(2:end, :);
    else
        Ucoef = mat{1,1} \ U;
    end
    Vcoef= mat{1,2} \ V;
    steps= 1:10;
    Aerr = zeros(size(steps));
    warning off
    for step = steps
        mesh = x1([1:step:480 481:720]);
        if nargout <= 3
            [Umesh, Vmesh, dUmesh, dVmesh] = UVremesh(Ucoef, Vcoef, x1, mesh, parity);
        else
            [Umesh, Vmesh, dUmesh, dVmesh, ddUmesh, ddVmesh] = UVremesh(Ucoef, Vcoef, x1, mesh, parity);
        end
        Uerr = U - Umesh;
        Verr = V - Vmesh;
        Uerr = rms(Uerr, 1) * S;
        Verr = rms(Verr, 1) * S;
        UerrOK = Uerr < epsfit;
        VerrOK = Verr < epsfit;
        Ufit(:, UerrOK) = Umesh(:, UerrOK);
        Vfit(:, VerrOK) = Vmesh(:, VerrOK);
        dUfit(:,UerrOK) = dUmesh(:,UerrOK);
        dVfit(:,VerrOK) = dVmesh(:,VerrOK);
        if nargout > 3
            ddUfit(:,UerrOK) = ddUmesh(:,UerrOK);
            ddVfit(:,VerrOK) = ddVmesh(:,VerrOK);
        end
        Ustep(UerrOK) = step;
        Vstep(VerrOK) = step;
        Afit = Ufit * S * (Vfit');
        Aerr(step) = rms(A - Afit, "all");
    end
    Unodes = zeros(size(Ustep));
    Vnodes = Unodes;
    for i = 1:length(Ustep)
        Unodes(i) = length([1:Ustep(i):480 481:720]);
        Vnodes(i) = length([1:Vstep(i):480 481:720]);
    end
    Afitx1 = dUfit * S * (Vfit');
    Afitx2 = Ufit * S * (dVfit');
    if nargout > 3
        Afitx1x1 = ddUfit * S * (Vfit');
        Afitx2x2 = Ufit * S * (ddVfit');
    end
    warning on
end

function N = SVDorder(U, S, V, eps)
    N = size(S, 1);
    for i=1:N
        if rms(U(:, i)) * S(i,i) * rms(V(:, i)) <= eps
            N = i;
            break
        end
    end
end

function [U, S, V] = SVDtrunc(U, S, V, N)
    U = U(:, 1:N);
    S = S(1:N,1:N);
    V = V(:, 1:N);
end

function [Ufit, Vfit, dUfit, dVfit, ddUfit, ddVfit] = UVremesh(Ucoef, Vcoef, oldmesh, newmesh, parity)
    odd = 1;
    mat2 = BS6mat(oldmesh, newmesh, 1, parity);
    Ufit= mat2{1,1} * Ucoef;
    Vfit= mat2{1,2} * Vcoef;
    mat3 = BS6mat(newmesh, newmesh, 1, parity);
    if parity == odd
        Ucoef= mat3{1,1}(2:end, :)\ Ufit(2:end, :);
    else
        Ucoef= mat3{1,1} \ Ufit;
    end
    Vcoef= mat3{1,2} \ Vfit;
    mat4 = BS6mat(newmesh, oldmesh, nargout/2, parity);
    Ufit= mat4{1,1} * Ucoef;
    Vfit= mat4{1,2} * Vcoef;
    if nargout > 2
        dUfit=mat4{2,1} * Ucoef;
        dVfit=mat4{2,2} * Vcoef;
        if nargout > 4
            ddUfit=mat4{3,1} * Ucoef;
            ddVfit=mat4{3,2} * Vcoef;
        end
    end
end

function mat = BS6mat(BSmesh, valmesh, ind, parity)
    f0 = @(x) 0;
    f1 = @(x) 1;
    % For the spline representation of omega and zeta, we do not use weights.
    Fconst = {f1, f0, f0, f0, f0, f0;
              f1, f0, f0, f0, f0, f0; };
    r = BSmesh(end)/BSmesh(end-1);
    odd = 1;
    if parity == odd
        mat = BS6_interp(BSmesh, BSmesh, valmesh, valmesh, Fconst, r, 1, 0, ind);
    else
        mat = BS6_interp2(BSmesh, BSmesh, valmesh, valmesh, Fconst, r, 1, 0, ind);
    end
end

function XYcoe = XYcoef(x1, x2, alpha, Chi20, AG)
    sr = (x1 .^ 2 + x2' .^ 2) .^ (1/2);
    sb = atan(x2' ./ x1);
    sr = reshape(sr, [], 1);
    sb = reshape(sb, [], 1);
    a  = 10;
    l1 = 50000;
    a2 = 100000;
    i1 = find(x1 > a - 1, 1) - 1;
    H = x1(end) - x1(end - 1);
    i2 = round((sqrt(x1(end) ^ 2 + x2(end) ^ 2) - x1(end)) / H) + 3;
    xr = x1(i1:end);
    xr = [xr; x1(end) + (1:i2)' * H];
    ord= 3;
    [Psi_rad, ~] = Dchi(xr, ord + 2, a, l1, a2, alpha, Chi20);
    XYcoe = Deri_polar_AGcoe(sr, sb, a, ord, Psi_rad, AG);
end

function fg = Leibni_prod(f, g, x, k)

    % compute d_x^k (f * g) using Leibniz rule. f{k+1}(x) for d_x^k f(x). Similar for g
    fg = zeros(size(x));

    for i = 0:k
        fg = fg + f{i + 1}(x) .* g{k - i + 1}(x) .* nchoosek(k, i);
    end

end

function chi = Assemble_chi(chi1, chi2, x, k)
    % assemble chi = chi2 * (1 - chi1) + chi1 using Leibniz rule

    chi = chi1(x, k);

    for i = 0:k

        if i == k
            chi = chi + nchoosek(k, i) * chi2(x, i) .* (1 - chi1(x, k - i));
        else
            % d_x 1 = 0
            chi = chi + nchoosek(k, i) * chi2(x, i) .* (- chi1(x, k - i));
        end

    end

end

function [Psi_rad, Chi] = Dchi(r, ord, a1, lam1, a2, alpha, Chi20)

    % Construct and estimate cutoff fucntion in Appendix D.1.3 in paper 2. Below section number is in paper 2
    %        chi = chi1 + (1 - chi1) * chi2 =   chi1 * (1 - chi2) + chi2
    % by first esitmating piecewise bounds of chi_i, then Lebiniz rule, and grid point + C^1, C^2 error bounds.
    % chi1 = chi10( (x - a1) /  lam1^(1/2) );  chi10 = chi_{rati} in App. D.1.2
    % chi2 = chi20( (x-a2)/ (9*a2)), chi20 is the exponential cutoff in App. D.1.1
    % chi2 = 0 for small x and near a1,    est_chi derivatives bound of chi,
    % Psi_rad functions for r^(2-alp) * chi,  est derivatives bound of Psi_rad

    itl = ~isa(r, 'double') || ~isa(alpha, 'double');

    % Chi10{i+1} = 1_{x>=0}d_x^i(x ^ 7 / (1 + x ^ 2) ^ (7/2))
    Chi10 = cell(ord + 1, 1);
    syms x;
    f = simplify(x ^ 7 / (1 + x ^ 2) ^ (7/2));
    for i = 1:ord + 1
        if i > 1
            f = simplify(diff(f, x, 1));
        end
        f1 = matlabFunction(f);
        % f1 = 0, for x < 0
        Chi10{i} = @(x) (x >= 0) .* f1(x);
    end

    % Put the parameters and variables to interval value
    if itl
        a1 = intval(a1);
        lam1 = intval(lam1);
        a2 = intval(a2);
        alpha = intval(alpha);
    end

    % involves k derivatives, formula for the rescaled cutoff function in App. D.1.3
    chi1 = @(x, k) Chi10{k + 1}((x - a1) / sqrt(lam1)) * lam1 ^ (-k / 2);
    chi2 = @(x, k) Chi20{k + 1}((x - a2) / (9 * a2)) * (9 * a2) ^ (-k);

    Chi = cell(ord + 1, 1);

    for k = 0:ord
        % compute k-th deriavtives of chi1 * ( 1 - chi2 ) + chi2
        Chi{k + 1} = @(x) Assemble_chi(chi1, chi2, x, k);
    end

    % compute Pow_fun{k+1} = d_x^k r^(2-alpha) = Ck * r^(2-alpha - k)
    Pow_fun = cell(ord + 1, 1);
    Psi_rad = cell(ord + 1, 1);

    for i = 0:ord

        if i == 0
            fac = 1;
        else
            fac = fac * (2 - alpha - i + 1);
        end

        Pow_fun{i + 1} = @(x) fac * x .^ (2 - alpha - i);

    end

    % build Psi_rad = r^(2-alpha) * chi, with derivatives

    for i = 0:ord
        Psi_rad{i + 1} = @(x) Leibni_prod(Chi, Pow_fun, x, i);
    end

    % At most return derivatives bound up to ord <= 7 since Chi is only C^(6, 1)
    if ord > 7
        ls = 1:7 + 1;
        Psi_rad = Psi_rad(ls);
    end

end

function coe = Deri_polar_AGcoe(r, b, a1, ord, Psi_rad, AG)

    % Given Psi_rad{i+1} derivatives for d_r^i A(r) and AG the coefficients in the formula below
    % Compute the coefficients C_{i,j,l}(r,b) below on x-y grid pts:
    %   d_x^i d_y^j (A(r) * B(b)) = sum AG_i,j,k,l / r^{i+j-k} * d_r^k A(r) d_b^l B(b)
    %                   = sum C_{i,j,l} (r, b) * d_b^l B(b),
    % The formula from Appendix C.3 is used to evaluate the far-field semi-analytic part.

    if isa(r, 'double') == 0 || isa(b, 'double') == 0
        itl = 1;
    else
        itl = 0;
    end

    nr = length(r);
    nb = length(b);

    r = reshape(r, nr, 1);
    b = reshape(b, nb, 1);

    if nr ~= nb
        error('Different length');
    end

    if ord >= size(AG, 1)
        error(' Exceed angular degree!');
    end

    coe = cell(ord + 1, ord + 1, ord + 1);

    if itl

        r_power = intval(zeros(nr, ord + 1));

        for i = 0:ord
            r_power(:, i + 1) = r .^ i;
        end

    else
        r_power = r .^ (0:ord);
    end

    % For r <= a1, the value = 0; A(r) is supported on fr > a1  Below restrict computation to lg
    lg = find(r > a1);

    Dpsi_rad = cell(ord + 1, 1);

    for k = 0:ord
        Dpsi_rad{k + 1} = Psi_rad{k + 1}(r(lg));
    end

    for deg = 0:ord

        for i = 0:deg
            j = deg - i;

            for l = 0:deg
                m = 0;

                coe{i + 1, j + 1, l + 1} = New_itl(zeros(nr, 1), itl);

                for k = 0:deg - l
                    ag = AG{i + 1, j + 1, k + 1, l + 1}(b(lg));

                    if isscalar(ag)
                        % AG{i + 1, j + 1, k + 1, l + 1} is a constant function
                        ag = ag .* ones(length(lg), 1);
                    end

                    m = m + ag ./ r_power(lg, deg - k + 1) .* Dpsi_rad{k + 1};

                end

                coe{i + 1, j + 1, l + 1}(lg) = m;

            end

        end

    end

end

function Psi1 = Deri_Psi1(n1, n2, g, BS_wg, ord, XYcoe)

    % Follow formulas in App.C.3.1 in paper 2 to evaluate 
    %  d_x^i d_y^j (A(r) * B(b)) = sum AG_i,j,k,l / r^{i+j-k} * d_r^k A(r) d_b^l B(b)
    %                   = sum C_{i,j,l} (r, b) * d_b^l B(b),     i+j <= ord <= 6,
    % XYcoe{i+1, j+1, l+1} for C_{i,j,l} on x-y grid pts is generated in Deri_polar_AGcoe,
    % Note: Matlab index starts from 1

    ord = min(ord, 6);

    Psi1 = cell(ord + 1, ord + 1);
    dg = cell(ord + 1, 1);

    % We represent g(b) = F(pi/2-b), F is spline odd at 0. See the end of Appendix C.1
    for i = 0:ord
        dg{i + 1} = (BS_wg{i + 1} * g) * (-1) ^ i;
    end

    for deg = 0:ord

        for i = 0:deg
            j = deg - i;
            m = 0;

            for l = 1:1 + deg
                m = m + XYcoe{i + 1, j + 1, l} .* dg{l};
            end

            Psi1{i + 1, j + 1} = reshape(m, n1, n2);

        end

    end

end

function D = Cell_2double(F)
    % put cell to double
    [s1, s2] = size(F);
    [s3, s4] = size(F{1, 1});
    D = NaN(s3, s4, s1, s2);

    for i = 1:s1
        for j = 1:s2
            if ~isempty(F{i, j})
                D(:, :, i, j) = F{i, j};
            end
        end
    end

end

function M = New_itl(M0, itl)
    % create new variables and change to intval value if itl = 1

    if itl
        M = intval(M0);
    else
        M = M0;
    end
end
