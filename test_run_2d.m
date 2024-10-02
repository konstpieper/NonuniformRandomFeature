is_octave = exist('OCTAVE_VERSION', 'builtin');
if is_octave
  pkg load optim;
end

addpath opt

rand('state', 1);
randn('state', 1);

%% width of the transition for the activation
delta = .025;

%% order of the activation
s = 1;

%% number of data points 
Npts = 1000;
p = setup_problem_NN_2d_s(delta, s, Npts, 'unif');

%% fancy plots
%p.label_interp = 'latex';

%figure(133);
%c_of_b = @(b) 1 ./ (b + sqrt(1 + b.^2));
%p.plot_forward(p, struct('x', c_of_b([1, -1]).*[1,-1].*[1;1]/sqrt(2), 'u', [1,1]), y_d);
%keyboard;

%%% helper functions
huber = @(t) (abs(t) <= 1/2) .* t.^2 + (abs(t) > 1/2) .* (abs(t) - 1/4);
sigma = @(t) (huber(t) + t)/2;

%% localized Gaussian
%f_d = @(x) exp(- sum((10*x).^2, 1) / 2);

%% 1D function
%f_d = @(x) sin(5 * sum([1; -sqrt(2)] .* x, 1));

%% oscillatory
%f_d = @(x) sin(10 * sum(x.^2, 1));

%% radial sinc
%f_d = @(x) sin(0.001 + 15 * sqrt(sum(x.^2, 1))) ./ (0.001 + 15 * sqrt(sum(x.^2, 1)));

%% (almost) nonsmooth
%kappa = 40000;
%softmax = @(t) max(t, [], 1) + 1/kappa * log(sum(exp(kappa*(t - max(t, [], 1))), 1)));
softmax = @(t) max(t, [], 1);
%f_d = @(x) softmax([zeros(1, size(x, 2)); x]);

%% Huberized norm
%f_d = @(x) huber(4*sqrt(sum((x).^2, 1)))/4;

%% lopsided Gaussian
%f_d = @(x) exp(-sqrt(sum(x.^2, 1)) - sigma(4 * x(1,:)));

%% misleading gradients
%f_d = @(x) cos(10 * sum([1; 0] .* x, 1)) + huber(7 * sum([0; 1] .* x));

%% misleading gradients cos
%f_d = @(x) sum(cos(5/2*sqrt(2)*pi*x), 1);
%f_d = @(x) sum(cos(5/sqrt(2)*pi* ([1,1; -1,1] * x)), 1);

%% misleading Hessian (harmonic function)
%f_d = @(x) sum(([1,1; -1,1] * x).^2 .* [1; -1], 1) + huber(5 * sum([1; 0] .* x));

%% anisotropic quadratic / Gaussian
Q = [ [1; 1]/sqrt(2), [-1; 1]/sqrt(2) ]; 
A = Q * diag([2, 1/4]) * Q';
xc = 0*[-.2;-.2];
%f_d = @(x) -1 + sum((x - xc) .* (A * (x - xc)), 1);
%f_d = @(x) exp( -20 * sum((x - xc) .* (A * (x - xc)), 1));

%% checkmark
%T = @(x) 4 * [x(1,:);
%	      2*(x(2,:) + 2*(1/6 - (huber(3 * x(1,:)) / 3)))];
%f_d = @(x) exp(- sum(T(x).^2, 1) / 2);
T = @(x) [x(1,:) + 2*(1/6 - (huber(3 * x(2,:))) / 3);
	  x(2,:)];
f_d = @(x) exp(- sum([8; 4].^2 .* T(x).^2, 1) / 2);

%% singular Green's
%f_d = @(x) - log(sqrt(sum(x.^2, 1) + 0.01^2));

%% Rosenbrock
sf = sqrt(2)*2;
%f_d = @(x) (sf*x(2,:) - (sf*x(1,:)).^2).^2 + (1 - sf*x(1,:)).^2/100;

%% Robot arm
transform = @(x) (sqrt(2) * x + 1) .* [1/2; pi];
extend = @(x) [ones(size(x)) .* [1/2; 0]; x];
%f_d = @(x) robot_arm(extend(transform(x)));

%% training and test data
y_d = f_d(p.xhat)';
y_d = y_d + 0.02*randn(size(y_d));

y_test = f_d(p.xhat_test)';

data = struct();
data.y_d = y_d;
data.y_test = y_test;

%% compute gradient and Hessian
[data.g_d, data.H_d] = compute_derivatives(f_d, p.xhat);

%% desired number of feature functions
N_target = 250; %floor(Npts/5);

%% hyperparameter for nonlocal weights
deltaW = 2*delta;

%[a, b] = build_weights(p.xhat);
[a, b] = sample_weights_uniform(p.dim, p.R, N_target);
%[a, b, A, B] = build_weights_grad(p.xhat, f_d, s);
%[a, b, A, B] = sample_weights_local(p.xhat, f_d, deltaW, s, N_target);
%[a, b, A, B] = sample_weights_nonlocal(p.xhat, f_d, deltaW, s, N_target);
%[a, b, A, B] = sample_weights_hyper(p, f_d, N_target);
%a = [a, A]; b = [b, B];

Nomega = size(b, 2);
omegas = stereo(a, b);

[coeff, alphal2] = lsqr_cval(p, omegas, data);
ul2 = struct('x', omegas, 'u', coeff');

figure(10);
p.plot_forward(p, ul2, y_d);
drawnow;

%% plot adaptive density

%[a, b] = sample_weights_TN(p.xhat, p.R, 30000);
[a, b, A, B] = sample_weights_local(p.xhat, f_d, deltaW, s, 30000);
%[a, b, A, B] = sample_weights_nonlocal(p.xhat, f_d, deltaW, s, 30000);
figure(98765);
plot_density(p, a, b, ones(1,size(b,2)));
%plot_kernel(p, stereo(a, b), [0;0]);

[a, b, A, B] = sample_weights_hyper(p, f_d, 30000);
%[a, b, A, B] = sample_weights_nonlocal(p.xhat, f_d, deltaW, s, 30000);
figure(987654);
plot_density(p, a, b, ones(1,size(b,2)));
%plot_kernel(p, stereo(a, b), [0;0]);

%% stochastic inner weight generation
alg_opts = struct();
alg_opts.plot_final = 0;
alg_opts.plot_every = 0;

% Number of sub-iterations (for residual based sampling)
Niter = 8;

% Number of trials to obtain robustness
Nsample = 10;

err_TN = zeros(Nsample, Niter);
err_AS = zeros(Nsample, Niter);
err_apf = zeros(Nsample, Niter);
err_apr = zeros(Nsample, Niter);
err_ngf = zeros(Nsample, Niter);
err_ngr = zeros(Nsample, Niter);
err_aphf = zeros(Nsample, Niter);
err_aphr = zeros(Nsample, Niter);
for nsam = 1:Nsample
  [u_TN, err_TN(nsam,:), N_TN] = iterative_approx(p, f_d, data, N_target, Niter, 'TN', alg_opts);
  [u_AS, err_AS(nsam,:), N_AS] = iterative_approx(p, f_d, data, N_target, Niter, 'AS', alg_opts);
  [u_apf, err_apf(nsam,:), N_apf] = iterative_approx(p, f_d, data, N_target, Niter, 'gradf', alg_opts);
  [u_apr, err_apr(nsam,:), N_apr] = iterative_approx(p, f_d, data, N_target, Niter, 'gradres', alg_opts);
  [u_ngf, err_ngf(nsam,:), N_ngf] = iterative_approx(p, f_d, data, N_target, Niter, 'ngf', alg_opts);
  %[u_ngr, err_ngr(nsam,:), N_ngr] = iterative_approx(p, f_d, data, N_target, Niter, 'ngres', alg_opts);
  [u_aphf, err_aphf(nsam,:), N_aphf] = iterative_approx(p, f_d, data, N_target, Niter, 'hyperf', alg_opts);
  %[u_aphr, err_aphr(nsam,:), N_aphr] = iterative_approx(p, f_d, data, N_target, Niter, 'hyperres', alg_opts);
end


%% optimal weight generation
alg_opts.plot_final = 0;
alg_opts.plot_every = 0;
alg_opts.max_step = 5000;
alg_opts.print_every = 50;
alg_opts.blocksize = 100;
alg_opts.sparsification = false;
alg_opts.optimize_x = true;
alg_opts.TOL = 1e-4;

deriv = 'derivative';
if s == 2
  deriv = 'Hessian';
  alpha0 = .004;
else
  deriv = 'gradient';
  alpha0 = .04;
end

gamma = 5;
phi = p.Phi(p, gamma);

alphas = alpha0 * (1/4).^(0:5);

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

%% plot density of trained network
figure(987653);
[a_opt, b_opt] = unstereo(u_opt{end}.x, p.kappa);
a_opt = [a_opt, -a_opt];
b_opt = [b_opt, -b_opt];
c_opt = [u_opt{end}.u, -u_opt{end}.u];
plot_density(p, a_opt, b_opt, c_opt);

%% plot accuracy graph
figure(12345)
set(gcf, 'Position', [0,0,800,600]);

hs = zeros(0);
hs(end+1) = plot_with_error(N_TN, err_TN, '-', 'blue', 'uniform');
hold on
hs(end+1) = plot_with_error(N_AS, err_AS, '--', 'green', 'active subspaces');
hs(end+1) = plot_with_error(N_apf, err_apf, ':', 'orange', deriv);
hs(end+1) = plot_with_error(N_apr, err_apr, '-.', 'orange', [deriv, ' residual']);
hs(end+1) = plot_with_error(N_ngf, err_ngf, ':', 'yellow', [deriv, ' nonlocal']);
%hs(end+1) = plot_with_error(N_ngr, err_ngr, '-.', 'yellow', [deriv, ' nonlocal residual']);
hs(end+1) = plot_with_error(N_aphf, err_aphf, '-.', 'purple', 'representation');
hs(end+1) = plot_with_error(N_phi, err_phi, '--', 'black', 'fully trained sparse');
l = legend(hs);
set(l, 'interpreter', p.label_interp);
legend('location', 'southwest')
axis('tight', 'tic')
ax = axis() .* exp(.1 * [-1,1,-1,1]);
%ax(4) = 1;
axis(ax)
set(gca, 'ticklabelinterpreter', p.label_interp)
set(gca, 'xminortick', 'on')
set(gca, 'yminortick', 'on')
grid on;
hold off;
