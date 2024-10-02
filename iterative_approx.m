function [u_ap, err_ap, N_ap] = iterative_approx(p, f_d, data, Ntarget, Niter, method, alg_opts)

Ndata = numel(data.y_d);
  
ul2 = p.u_zero;
No_tot = 0;

u_ap = cell(Niter,1);
N_ap = zeros(1,Niter);
%err_ap = cell(Niter,1);
err_ap = zeros(1,Niter);
alphas = zeros(1,Niter);

Nomegas = round(logspace(log10(4), log10(Ntarget), Niter));

tic;
for kni = 1:Niter
  %Nomega = round(Ntarget * kni/Niter) - No_tot;
  Nomega = Nomegas(kni) - No_tot;

  %% hyperparameter for weight generation
  deltaW = 2*p.delta;
  
  B = []; A = [];
  if strcmp(method, 'TN')
    [a, b] = sample_weights_uniform(p.dim, p.R, Nomega);
  elseif strcmp(method, 'AS')
    [a, b] = sample_weights_AS(p.xhat, f_d, p.R, p.s, Nomega);
  elseif strcmp(method, 'gradf')
    [a, b, A, B] = sample_weights_local(p.xhat, f_d, deltaW, p.s, Nomega);
  elseif strcmp(method, 'gradres')
    res = @(x) f_d(x) - p.K(p, x, ul2)';
    [a, b, A, B] = sample_weights_local(p.xhat, res, deltaW, p.s, Nomega);
  elseif strcmp(method, 'ngf')
    [a, b, A, B] = sample_weights_nonlocal(p.xhat, f_d, deltaW, p.s, Nomega);
  elseif strcmp(method, 'ngres')
    res = @(x) f_d(x) - p.K(p, x, ul2)';
    [a, b, A, B] = sample_weights_nonlocal(p.xhat, res, deltaW, p.s, Nomega);
  elseif strcmp(method, 'hyperf')
    [a, b, A, B] = sample_weights_hyper(p, f_d, Nomega);
  elseif strcmp(method, 'hyperres')
    res = @(x) f_d(x) - p.K(p, x, ul2)';
    [a, b, A, B] = sample_weights_hyper(p, res, Nomega);
  end

  perm = randperm(size(b,2));
  a = a(:,perm(1:Nomega));
  b = b(:,perm(1:Nomega));

  if kni == 1 && !isempty(B)
    a = [A, a];
    b = [B, b];
  end

  omegas = stereo(a, b);

  ul2.x = [ul2.x, omegas];
  No_tot = size(ul2.x, 2);

  [coeff, alphas(kni)] = lsqr_cval(p, ul2.x, data);
  ul2.u = coeff';

  if get_field_default(alg_opts, 'plot_every', false);
    figure(111);
    p.plot_forward(p, ul2, data.y_d);
    drawnow;
  end

  [err, err_test] = error_cval(p, ul2, data.y_d, data.y_test);

  u_ap{kni} = ul2;
  N_ap(kni) = No_tot;
  %err_ap{kni} = [err, err_test];
  err_ap(kni) = err_test;
end

timer = toc();

fprintf('iterative_approx: %8s, err_train=%1.1e, err_test=%1.1e, N=%d, %1.1f s\n', ...
            method, err, err_test, No_tot, timer);

if get_field_default(alg_opts, 'plot_final', false);
  figure(111);
  p.plot_forward(p, ul2, data.y_d);
  drawnow;
end

end
