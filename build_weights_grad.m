function [a_cand, b_cand, A, B] = build_weights_grad(xhat, f, s)

  dim = size(xhat, 1);
  Nx = size(xhat, 2);
  gf = zeros(dim, Nx);

  f_x = f(xhat);
  tau = sqrt(eps);
  for d = 1:dim
    gf(d,:) = (f(xhat + tau * ((1:dim) == d)') - f(xhat)) / tau;
  end

  Agf = sum(gf, 2) / Nx;
  
  gf_norm = sqrt(sum(gf.^2, 1));

  a_hat = gf ./ gf_norm;
  a_rand = rand(dim, sum(gf_norm == 0));
  a_rand = a_rand ./ sqrt(sum(a_rand.^2, 1));
  a_hat(:, gf_norm == 0) = a_rand;

  b_hat = - sum(xhat .* a_hat, 1);

  a_cand = [a_hat, -a_hat];
  b_cand = [b_hat, -b_hat];

  if s == 1
    A = [zeros(dim, 1)];
    B = [1];
  else
    A = [Agf/norm(Agf,2), -Agf/norm(Agf,2), zeros(dim, 1)];
    B = [0, 0, 1];
  end
  
end
