is_octave = exist('OCTAVE_VERSION', 'builtin');
if is_octave
  pkg load optim;
end

addpath opt

rand('state', 1);
randn('state', 1);

%% width of the transition for the activation
delta = .0125;

%% order of the activation
s = 1;

%% number of data points 
Npts = 1000;
p = setup_problem_NN_1d_s(delta, s, Npts, 'unif');

%% fancy plots
%p.label_interp = 'latex';

%% helper functions
huber = @(t) (abs(t) <= 1/2) .* t.^2 + (abs(t) > 1/2) .* (abs(t) - 1/4);
sigma = @(t) (huber(t) + t)/2;

%% localized Gaussian
f_d = @(x) exp(- sum((10*x).^2, 1) / 2);

%% flat hat
%f_d = @(x) max(0, min(1, 4*(1-2*abs(x))));

%f_d = @(x) exp(- 10*sqrt(sum(1e-4 + x.^2, 1)));
%f_d = @(x) max(2*(1- 2*x.^2), (1 - x)/2)

y_d = f_d(p.xhat)';
y_d = y_d + 0.02*randn(size(y_d));

y_test = f_d(p.xhat_test)';

data = struct();
data.y_d = y_d;
data.y_test = y_test;

%% compute gradient and Hessian
[data.g_d, data.H_d] = compute_derivatives(f_d, p.xhat);

%% desired number of feature functions
N_target = 50; %floor(Npts/10);

[a, b] = sample_weights_uniform(p.dim, 1, N_target);
Nomega = size(b, 2);
omegas = stereo(a, b);

[coeff, alphal2] = lsqr_cval(p, omegas, data);
ul2 = struct('x', omegas, 'u', coeff');

figure(10);
p.plot_forward(p, ul2, y_d);
drawnow;

%% stochastic inner weight generation
alg_opts = struct();
alg_opts.plot_every = 0;
alg_opts.plot_final = 0;

% Number of sub-iterations (for residual based sampling)
Niter = 8;

% Number of trials to obtain robustness
Nsample = 10;

err_TN = zeros(Nsample, Niter);
err_apf = zeros(Nsample, Niter);
err_apr = zeros(Nsample, Niter);
err_ngf = zeros(Nsample, Niter);
err_ngr = zeros(Nsample, Niter);
err_aphf = zeros(Nsample, Niter);
%err_aphr = zeros(Nsample, Niter);
for nsam = 1:Nsample
  [u_TN, err_TN(nsam,:), N_TN] = iterative_approx(p, f_d, data, N_target, Niter, 'TN', alg_opts);
  [u_apf, err_apf(nsam,:), N_apf] = iterative_approx(p, f_d, data, N_target, Niter, 'gradf', alg_opts);
  [u_apr, err_apr(nsam,:), N_apr] = iterative_approx(p, f_d, data, N_target, Niter, 'gradres', alg_opts);
  [u_ngf, err_ngf(nsam,:), N_ngf] = iterative_approx(p, f_d, data, N_target, Niter, 'ngf', alg_opts);
  [u_ngr, err_ngr(nsam,:), N_ngr] = iterative_approx(p, f_d, data, N_target, Niter, 'ngres', alg_opts);
  [u_aphf, err_aphf(nsam,:), N_aphf] = iterative_approx(p, f_d, data, N_target, Niter, 'hyperf', alg_opts);
  %[u_aphr, err_aphr(nsam,:), N_aphr] = iterative_approx(p, f_d, data, N_target, Niter, 'hyperres');
end

%% optimal weight generation
alg_opts.max_step = 10000;
alg_opts.plot_every = 0;
alg_opts.plot_final = 0;
alg_opts.print_every = 50;
alg_opts.blocksize = 100;
alg_opts.sparsification = true;
alg_opts.optimize_x = true;
alg_opts.TOL = 1e-4;

deriv = 'derivative';
if s == 2
  deriv = 'Hessian';
  alpha0 = .01;
else
  deriv = 'gradient';
  alpha0 = .1;
end

gamma = 10;
phi = p.Phi(p, gamma);

alphas = alpha0 * (1/4).^(0:4);

u_opt = cell(length(alphas),1);
alg_out = cell(length(alphas),1);
N_phi = zeros(1,length(alphas));
err_phi = zeros(1,length(alphas));

uinit = p.u_zero;

timer = 0;

for n = 1:length(alphas)
  alpha = alphas(n);
  
  alg_opts.u0 = uinit;
  %[u_opt{n}, alg_out{n}] = PDAPmultisemidiscrete(p, y_d, alpha, phi, alg_opts);
  [u_opt{n}, alg_out{n}] = solve_TV_CGNAP(p, y_d, alpha, phi, alg_opts);

  uinit = u_opt{n};
  
  figure(3);
  p.plot_adjoint(p, u_opt{n}, p.obj.dF(p.K(p, p.xhat, u_opt{n})-y_d), alpha)
  figure(4);
  p.plot_forward(p, u_opt{n}, y_d)
  drawnow;

  N_phi(n) = size(u_opt{n}.x, 2);
  [err, err_test] = error_cval(p, u_opt{n}, y_d, y_test);
  err_phi(n) = err_test;

  timer = timer + alg_out{n}.tics(end);
end
fprintf('training time: %d s\n', timer);

%% plot accuracy graph
figure(12345)
set(gcf, 'Position', [0,0,800,600]);

hs = zeros(0);
hs(end+1) = plot_with_error(N_TN, err_TN, '-', 'blue', 'uniform');
hold on
hs(end+1) = plot_with_error(N_apf, err_apf, ':', 'orange', deriv);
%hs(end+1) = plot_with_error(N_apr, err_apr, '-.', 'orange', [deriv, ' residual']);
%hs(end+1) = plot_with_error(N_ngf, err_ngf, ':', 'yellow', [deriv, ' nonlocal']);
%hs(end+1) = plot_with_error(N_ngr, err_ngr, '-.', 'yellow', [deriv, ' nonlocal residual']);
hs(end+1) = plot_with_error(N_aphf, err_aphf, '-.', 'purple', 'representation');
hs(end+1) = plot_with_error(N_phi, err_phi, '--', 'black', 'fully trained sparse');
l = legend(hs);
set(l, 'interpreter', p.label_interp);
axis('tight', 'tic')
ax = axis() .* exp(.1 * [-1,1,-1,1]);
ax(4) = 1;
axis(ax)
set(gca, 'ticklabelinterpreter', p.label_interp)
set(gca, 'xminortick', 'on')
set(gca, 'yminortick', 'on')
grid on;
hold off;
