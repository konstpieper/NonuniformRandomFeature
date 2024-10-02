function [coeff, alpha] = lsqr_cval(p, omegas, data)

Nomega = size(omegas, 2);

Nalpha = 16;
alphas = logspace(1, -4, Nalpha);

%% prepare matrizes for ridge regression / least squares
[Khat, dKhat] = p.k(p, p.xhat, omegas);
obs = p.obj.dF(Khat'*data.y_d);
H = Khat' * p.obj.ddF(0) * Khat;

%% for lambda > 0, add gradient data to the fit
lambda = 0.;
if lambda > 0;
  for d = 1:size(dKhat, 3)
    obs = obs + lambda * p.obj.dF(dKhat(:,:,d)'*data.g_d(d,:)');
    H = H + lambda * dKhat(:,:,d)' * p.obj.ddF(0) * dKhat(:,:,d);
  end
end

IN = eye(Nomega)/Nomega;

err = zeros(Nalpha, 1);
err_test = zeros(Nalpha, 1);

%% grid search for the regularization parameter
for cni = 1:Nalpha
  alpha = alphas(cni);
  coeff = (alpha*IN + H) \ obs;

  [err(cni), err_test(cni)] = error_cval(p, struct('x', omegas, 'u', coeff'), data.y_d, data.y_test);
end

%% minimum validation error
[minerr, mini] = min(err_test);
%ind = mini;

%% take the largest regularizer that comes within 5% of optimal
ind = min(find(err_test <= 1.05 * minerr));

alpha = alphas(ind);
coeff = (alpha*IN + H) \ obs;

end
