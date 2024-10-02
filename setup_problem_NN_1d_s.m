function p = setup_problem_NN_1d_s(delta, s, Npts, data_dist)

p = struct();

%% order and smoothness parameter
p.s = s;
p.delta = delta;

% scalar problem
p.N = 1;

% input dimension
p.dim = 1;

% size of the box
p.L = 1;

% relevant ball for the input weights
p.R = p.L;
% radii for the parametrization a(x), b(x)
RO = p.R + sqrt(1 + p.R^2);
rO = -p.R + sqrt(1 + p.R^2);
p.Omega = [rO, RO];

%% set the cost for the "bias" b
p.kappa = 1e-1 / p.R;

% zero measure
p.u_zero = struct('x', zeros(p.dim,0), 'u', zeros(p.N,0));

%% observation set
if strcmp(data_dist, 'gauss')
  p.xhat = p.L * randn(1, Npts) / 3;
  p.xhat_test = p.L * randn(size(p.xhat)) / 3;
else
  p.xhat = linspace(-p.L, p.L, Npts);
  p.xhat_test = (-1 + 2*rand(size(p.xhat))) * p.L;
end

%figure(1234)
%plot(p.xhat(1,:), p.xhat(2,:), 'ko', p.xhat_test(1,:), p.xhat_test(2,:), 'bx')

%% kernel
p.K = @K;
p.k = @kernel;
p.plot_forward = @plot_forward;

%% stuff for optimization
p.Phi = @Phi;
p.obj = Tracking(p);

%% functions for PDAP algorithm
p.Ks = @Ks;
p.sample_param = @sample_param;
p.find_max = @find_max;
p.optimize_u = @optimize_u;
p.optimize_xu = @optimize_xu;

p.plot_adjoint = @plot_adjoint;
p.postprocess = @postprocess;

p.k_dx = @kernel_dx;
p.k_oc = @kernel_of_choice;

%% set to 'latex' to obtain nice labels (slower than 'tex')
p.label_interp = 'tex';

end


function [Ku, dKu] = K(p, xhat, u)

if (nargout > 1)
  [k, dk] = kernel(p, xhat, u.x);
else
  k = kernel(p, xhat, u.x);
end

Ku = k * u.u(:);

if (nargout > 1)
  dKu = zeros(p.dim, size(xhat,2));
  for j = 1:p.dim
    dKu(j,:) = dk(:,:,j) * u.u(:);
  end
end

end

function [Ksy, dKsy] = Ks(p, x, xhat, y)

if (nargout > 1)
  [k, dk] = kernel(p, xhat, x);
else
  k = kernel(p, xhat, x);
end

Ksy = k' * y;

if (nargout > 1)
  dKsy = zeros(p.dim, size(x,2));
  for j = 1:p.dim
    dKsy(j,:) = dk(:,:,j)' * y;
  end
end

end

function [k, dk] = kernel(p, xhat, x)

Nx = size(x, 2);
Nxh = size(xhat, 2);

X = zeros(Nxh, Nx);
Xhat = zeros(Nxh, Nx);
[X(:,:), Xhat(:,:)] = meshgrid(x(1,:), xhat(1,:));

%% proceed as in unstereo.m applied to om=X
x2 = X.^2;
r = sqrt(4 * x2 + p.kappa * (1 - x2).^2);

%% b = (1 - x2) ./ r;
%% a =    2 * x ./ r;  
xxhat = X.*Xhat;
%% y = a*xhat + b = yt / r
y = (2*xxhat + (1 - x2)) ./ r;

%% smoothing parameter
delta = p.delta;

if nargout == 1;
  kd = sigmoid(y / delta, p.s);
  k = kd * delta.^(p.s-1);
else 
  [kd, dkd] = sigmoid(y / delta, p.s);
  k = kd * delta.^(p.s-1);

  %% dydx = (dytdx - y*drdx) / r
  drdx = (4 - 2 * p.kappa * (1 - x2)) .* X ./ r;
  dydx = (2*Xhat - 2*X - y .* drdx) ./ r;

  dk = dkd .* dydx * delta.^(p.s-2);
end

end

function [k, dk] = kernel_dx(p, xhat, x)

Nx = size(x, 2);
Nxh = size(xhat, 2);

X = zeros(Nxh, Nx);
Xhat = zeros(Nxh, Nx);
[X(:,:), Xhat(:,:)] = meshgrid(x(:), xhat(:));

%% proceed as in unstereo.m applied to om=X
x2 = X.^2;
r = sqrt(4 * x2 + p.kappa * (1 - x2).^2);

%% b = (1 - x2) ./ r;
%% a =    2 * x ./ r;  
xxhat = X.*Xhat;
%% y = a*xhat + b = yt / r
y = (2*xxhat + (1 - x2)) ./ r;

% smoothing parameter
delta = p.delta;

[kd, dkd] = sigmoid(y / delta, p.s);
k = kd * delta.^(p.s-1);

a = 2 * X ./ r;
dkdxh = dkd .* a * delta.^(p.s-2);
dk = dkdxh;

end

function k = kernel_of_choice(p, xhat, x, eta)

Nx = size(x, 2);
Nxh = size(xhat, 2);

X = zeros(Nxh, Nx);
Xhat = zeros(Nxh, Nx);
[X(:,:), Xhat(:,:)] = meshgrid(x(1,:), xhat(1,:));

%% proceed as in unstereo.m applied to om=X
x2 = X.^2;
r = sqrt(4 * x2 + p.kappa * (1 - x2).^2);

%% b = (1 - x2) ./ r;
%% a =    2 * x ./ r;  
xxhat = X.*Xhat;
%% y = a*xhat + b = yt / r
y = (2*xxhat + (1 - x2)) ./ r;

% smoothing parameter
delta = p.delta;

kd = eta(y / delta);
k = kd * delta;

end

function phi = Phi(p, gamma)

phi = struct();
phi.gamma = gamma;

if gamma == 0
    phi.phi = @(t) t;
    phi.dphi = @(t) ones(size(t));
    phi.ddphi = @(t) zeros(size(t));
    phi.inv = @(y) y;

    phi.prox = @(sigma, g) max(g - sigma, 0);
else
    th = 1/2;
    gam = gamma/(1-th);

    phi.phi = @(t) th * t + (1-th) * log(1 + gam * t) / gam;
    phi.dphi = @(t) th + (1-th) ./ (1 + gam * t);
    phi.ddphi = @(t) - (1-th) * gam ./ (1 + gam * t).^2;
    phi.inv = @(y) y / th; % not the inverse but an upper bound for the inverse

    phi.prox = @(sigma, g) (1/2)*max((g - sigma*th - 1/gam) + sqrt( (g - sigma*th - 1/gam)^2 + 4*(g - sigma)/gam), 0);

    %phi.phi = @(t) log(1 + gamma * t) / gamma;
    %phi.dphi = @(t) 1 ./ (1 + gamma * t);
    %phi.ddphi = @(t) - gamma ./ (1 + gamma * t).^2;
    %phi.inv = @(y) (exp(gamma * y) - 1) / gamma;

    %phi.prox = @(sigma, g) max((g - 1/gamma)/2 + sqrt( ((g - 1/gamma)/2)^2 + (g - sigma)/gamma), 0);
end

end

function obj = Tracking(p)

Nx = numel(p.xhat);
  
obj.F = @(y) 1/(2*Nx) * norm(y)^2;
obj.dF = @(y) 1/Nx * y;
obj.ddF = @(y) 1/Nx;

%p = 5;
%obj.F = @(y) 1/(p) * norm(y, p)^p;
%obj.dF = @(y) 1 * sign(y).*abs(y).^(p-1);
%obj.ddF = @(y) 1 * spdiags((p-1)*abs(y).^(p-2), 0, Nx, Nx);

end

function u_pp = postprocess(p, u, pp_radius)

  %% take a given discrete measure u and lump together Dirac delta functions
  %% that are less than pp_radius apart
  
  [x, perm] = sort(u.x);
  u = u.u(perm);
  
  cut = [0, find(diff(x) > pp_radius), length(x)];
  U = zeros(1, length(cut)-1);
  X = zeros(1, length(cut)-1);
  for cp = 1:length(cut)-1
    range = cut(cp)+1:cut(cp+1);
    U(1,cp) = sum(u(1,range));
    X(1,cp) = sum(x(1,range).*abs(u(1,range))) / sum(abs(u(1,range)));
  end
  
  u_pp = struct('x', X, 'u', U);

end

function plot_forward(p, u, y_d)

  %% plot the measure and the function and its approximation
  pO = [-1,1] * p.Omega(2);

  knot_p = @(x) min(10, max(-10, -(1 - x.^2)./(2*x + eps)));
  
  %% plot measure

  subplot(2,1,1);

  ux = u.x(1,u.u ~= 0);
  uu = u.u(1,u.u ~= 0);

  xi = [ux; ux];
  coef = uu ./ (sign(ux)).^p.s;
  yi = [zeros(size(coef)); coef];
  plot(knot_p(xi), yi, 'k', 'LineWidth', 1);
  hold on;
  plot(pO, [0,0], 'k--', 'LineWidth', 1);
  plot(knot_p(ux), coef, 'ko', 'LineWidth', 3, 'MarkerSize', 3);
  set(gca, 'FontSize', 12);
  axis([-p.L, p.L, min([coef,-.01])*1.2, max([coef,.01])*1.2]);
  xlabel('$x_n = -b_n / a_n$', 'interpreter', p.label_interp)
  if mod(p.s, 2) == 1
    ylabel('$c_n / {\rm sign}(a_n)$', 'interpreter', p.label_interp)
  else
    ylabel('$c_n$', 'interpreter', p.label_interp)
  end
  set(gca, 'ticklabelinterpreter', p.label_interp)
  hold off;

  %% plot function

  subplot(2,1,2);

  t = linspace(1.2*min(p.xhat), 1.2*max(p.xhat), 1000);
  plot(p.xhat, y_d, '--', 'LineWidth', 2, 'Color', [0 0.4470 0.7410]);
  hold on;
  Kxu = reshape(p.K(p, knot_p(ux), u), [p.N, length(ux)]);
  plot(knot_p(ux), Kxu, 'rx', 'LineWidth', 2, 'MarkerSize', 10, 'Color', [0.8500 0.3250 0.0980]);
  Ku = reshape(p.K(p, t, u), [p.N, length(t)]);
  plot(t, Ku, 'k', 'LineWidth', 1);
  axis([min(t), max(t), min(-.1,min(y_d))*1.2, max(.1,max(y_d))*1.2])
  set(gca, 'FontSize', 12);
  xlabel('$x$', 'interpreter', p.label_interp)
  ylabel('$y$', 'interpreter', p.label_interp)
  set(gca, 'ticklabelinterpreter', p.label_interp)
  hold off;

end

function plot_adjoint(p, u, y, alpha, pp_radius)

  %% plot the measure and the dual variable corresponding to y
  
  if nargin == 5
    u = postprocess(p, u, pp_radius);
  end

  pO = [-1,1] * p.Omega(2);
  t = [-1000, linspace(pO(1), pO(2), 1000), 1000];

  knot_p = @(x) min(10, max(-10, -(1 - x.^2)./(2*x + eps)));
  
  %% plot measure

  subplot(2,1,1);

  xi = [u.x; u.x];
  yi = [zeros(size(u.u)); u.u];
  plot(xi, yi, 'k', 'LineWidth', 1);
  hold on;
  plot(pO, [0,0], 'k--', 'LineWidth', 1);
  plot(u.x, u.u, 'ko', 'LineWidth', 3, 'MarkerSize', 3);
  set(gca, 'FontSize', 12);
  hold off;
  axis([pO, min(-.01,min(u.u))*1.2, max(.01,max(u.u))*1.2]);

  %% plot dual variable

  subplot(2,1,2);

  Ksy = reshape(p.Ks(p, t, p.xhat, y), [p.N, length(t)]);
  plot(t, Ksy, 'k', 'LineWidth', 2);
  hold on;
  Ksysupp = reshape(p.Ks(p, u.x, p.xhat, y), [p.N, length(u.x)]);
  plot(u.x, Ksysupp, 'ko', 'LineWidth', 1, 'MarkerSize', 5);
  plot(pO, [alpha, alpha], 'k--', 'LineWidth', 1);
  plot(pO, [-alpha, -alpha], 'k--', 'LineWidth', 1);
  axis([pO, min(-alpha,min(Ksy))*1.2, max(alpha,max(Ksy))*1.2])
  set(gca, 'FontSize', 12);
  hold off;

end

function u_opt = optimize_u(p, y_d, alpha, phi, u)

Kred = p.k(p, p.xhat, u.x);
ured = u.u(:);

% Solve subproblem with SSN
%ured = SSN(p, Kred, y_d, alpha, phi, ured, p.N);
%ured = SSN_TR(p, Kred, y_d, alpha, phi, ured, p.N);
ured = SSN_simple(p, Kred, y_d, alpha, phi, ured);

u_opt = u;
u_opt.u = ured';

end

function xrand = sample_param(p, Ntarget)

%% generate Ntarget random parameters in the desired parameter set
[a, b] = sample_weights_uniform(p.dim, p.R, Ntarget);
xrand = stereo(a, b);
  
end

function xmax = find_max(p, y, x0)

TOL = 1e-6;

% initial guesses
Nguess = 50;

if size(x0,2) > Nguess/2
  ii = randperm(size(x0,2));
  x0 = x0(:,ii(1:floor(Nguess/2)));
end
Nguess = Nguess - size(x0,2);

randomx = sample_param(p, Nguess);
x0 = [zeros(p.dim,1), randomx, x0];
 
y_norm = y/norm(y);
j  = @(x) optfun_vJ(p, y_norm, x);

ng = 100;
iter = 0;
while ng > 1e4*TOL && iter < 10
  
  %Hpat = kron(speye(size(x0,2)),[1,1;1,1]);
  opts = optimset('Algorithm', 'quasi-newton', 'GradObj', 'on', 'Display', 'off', ...
                  'TolFun', 1e-30, 'TolX', TOL, 'MaxIter', 500, 'MaxFunEvals', 10000);
  %opts = optimset('Algorithm', 'trust-region', 'GradObj', 'on', 'HessPattern', Hpat, 'Display', 'off', ...
  %                'TolFun', 1e-30, 'TolX', TOL, 'MaxIter', 500, 'MaxFunEvals', 10000);
    
  [xmax, ~, cvg] = fminunc(j, x0(:), opts);
  [jmax, g] = j(xmax);
  ng = norm(g);

  xmax = reshape(xmax, size(x0));
  u_pp = p.postprocess(p, struct('x', xmax, 'u', ones(1,size(xmax,2))), 1e2*TOL);
  
  xmax = u_pp.x;
  %Ksy = Ks(p, xmax, p.xhat, y);
  %xmax(:,Ksy > 0) = [];

  fprintf('\tpos: val: %f -> %f, |g|=%f (%i)\n', j(x0(:)), jmax, ng, cvg);
  x0 = xmax;

  %check_gradient(p, xmax(:), j);
  iter = iter + 1;
end

%figure(667)
%p.plot_adjoint(p, p.u_zero, y, 1);
%hold on;
%plot(xmax(1,:), xmax(2,:), 'xk');
%hold off;

end

function [j, g] = optfun_vJ(p, y, x)
  Nx = length(x)/p.dim;
  x = reshape(x, p.dim, Nx);
  [Ksy, dKsy] = Ks(p, x, p.xhat, y);
  Ksy = reshape(Ksy, [1, Nx]);
  dKsy = reshape(dKsy, [p.dim, Nx]);
  normK2 = sum(abs(Ksy).^2, 1);
  j = - sum(normK2, 2) / 2;
  g = - reshape(real(Ksy.*conj(dKsy)), [p.dim*Nx,1]);
end

%% for positive measures
%function [j, g] = optfun_vJ(p, y, x)
%  Nx = length(x)/p.dim;
%  x = reshape(x, p.dim, Nx);
%  [Ksy, dKsy] = Ks(p, x, p.xhat, y);
%  Ksy = reshape(Ksy, [1, Nx]);
%  dKsy = reshape(dKsy, [p.dim, Nx]);
%  normK2 = sum(Ksy, 1);
%  j = sum(normK2, 2) / 2;
%  g = reshape(dKsy, [p.dim*Nx,1]);
%end

function check_gradient(p, x, j, dj)

  if nargin > 3
    gx = dj(x);
  else
    [~, gx] = j(x);
  end
  
  dx = rand(size(x));
  tau = 2.^(0:-1:-30);
  err = zeros(size(tau));
  fin_diff = zeros(size(tau));
  for ntau = 1:length(tau)
    jp = j(x + tau(ntau)*dx);
    jm = j(x - tau(ntau)*dx);
    fin_diff(ntau) = (jp - jm)/(2*tau(ntau));
    err(ntau) = abs(gx'*dx - fin_diff(ntau));
  end
  figure(666);
  loglog(tau, err);
  drawnow;
  
end

function u_opt = optimize_xu(p, y_d, alpha, phi, u)

[dim, Nd] = size(u.x);
  
% initial guesses
xu0 = [u.x; u.u];

% upper and lower bounds and (fixed) signum
ugeq0 = (u.u >= 0);
ul = -Inf(1,Nd); ul(ugeq0) = 0;
uu = Inf(1,Nd); uu(~ugeq0) = 0;
xul = [-Inf(dim,Nd); ul];
xuu = [Inf(dim,Nd); uu];

xu0 = xu0(:);
xul = xul(:);
xuu = xuu(:);

signu = -ones(1,Nd);
signu(ugeq0) = 1;

is_octave = exist('OCTAVE_VERSION', 'builtin');

if ~is_octave
    %% MATLAB
    j = @(xu) optfun_vJxu(p, y_d, alpha, phi, signu, xu);
  
    opts = optimset('Algorithm', 'trust-region-reflective', ...
                    'GradObj', 'on', 'Hessian', 'off', ...
                    'Display', 'iter', 'DerivativeCheck', 'off', 'FinDiffType', 'central', ...
                    'TolFun', 1e-30, 'TolX', 1e-4, 'MaxIter', 200, 'MaxFunEvals', 1000);

    xumin = fmincon(j, xu0, [], [], [], [], xul, xuu, [], opts);
    [jmin, g] = j(xumin);
    inactive = (xumin < xuu) & (xumin > xul);
    fprintf('\toptimized points and coeff: val: %f -> %f, |g|=%f\n', j(xu0), jmin, norm(g(inactive)));

    %check_gradient(p, xumin, j);    
else
%% OCTAVE
    j  = @(xu) optfun_vxu(p, y_d, alpha, phi, signu, xu);
    dj = @(xu) optfun_Jxu(p, y_d, alpha, phi, signu, xu);

    opts = optimset('Algorithm', 'lm_feasible', 'lb', xul, 'ub', xuu, ...
                   'TolFun', 1e-14, 'objf_grad', dj, 'MaxIter', 200 );

    [xumin, minval, cvg] = nonlin_min(j, xu0, opts);

    inactive = (xumin < xuu) & (xumin > xul);
    g = dj(xumin);
    ng = norm(g(inactive));
    if cvg > 0
      fprintf('\toptimized points: val: %f -> %f, |g|=%f, status=%i\n', j(xu0), minval, ng, cvg)
    else
      fprintf('\tfailed opt. points: val: %f -> %f, |g|=%f, status=%i\n', j(xu0), minval, ng, cvg)
    end

    %check_gradient(p, xumin, j, dj);
end

xumin = reshape(xumin, dim+1, Nd);

u_opt = u;
u_opt.x = xumin(1:dim,:);
u_opt.u = xumin(dim+1,:);


end

function [j,g] = optfun_vJxu(p, y_d, alpha, phi, signu, xu)

Nd = length(xu) / (p.dim+1);
xu = reshape(xu, p.dim+1, Nd);

x = xu(1:p.dim,:);
u = xu(p.dim+1,:);

[K, dK] = kernel(p, p.xhat, x);
y = K*u';
j = p.obj.F(y - y_d) + alpha*sum(phi.phi(signu.*u));

if nargout > 1
    gF = p.obj.dF(y - y_d);

    g_x = zeros(p.dim, Nd);
    for i = 1:p.dim
        g_x(i,:) = u.*reshape(dK(:,:,i)'*gF, 1, Nd);
    end
    g_u = gF'*K + alpha*(signu.*phi.dphi(signu.*u));

    g = [g_x; g_u];
    g = g(:);
end

end

function j = optfun_vxu(p, y_d, alpha, phi, signu, xu)

Nd = length(xu) / (p.dim+1);
xu = reshape(xu, p.dim+1, Nd);

x = xu(1:p.dim,:);
u = xu(p.dim+1,:);

K = kernel(p, p.xhat, x);
y = K*u';
j = p.obj.F(y - y_d) + alpha*sum(phi.phi(signu.*u));

end

function g = optfun_Jxu(p, y_d, alpha, phi, signu, xu)

Nd = length(xu) / (p.dim+1);
xu = reshape(xu, p.dim+1, Nd);

x = xu(1:p.dim,:);
u = xu(p.dim+1,:);

[K, dK] = kernel(p, p.xhat, x);
y = K*u';

gF = p.obj.dF(y - y_d);

g_x = zeros(p.dim, Nd);
for i = 1:p.dim
  g_x(i,:) = u.*reshape(dK(:,:,i)'*gF, 1, Nd);
end
g_u = gF'*K + alpha*(signu.*phi.dphi(signu.*u));

g = [g_x; g_u];
g = g(:);

end
