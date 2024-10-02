function [a_cand, b_cand] = sample_weights_AS(xhat, f_d, radius, s, N_target)

  dim = size(xhat, 1);
  Nx = size(xhat, 2);

  [gf, Hf] = compute_derivatives(f_d, xhat);

  if s == 1
    [U, S, V] = svd(gf, 'econ');
  elseif s == 2
    [U, S, V] = svd(reshape(Hf, dim, dim*Nx), 'econ');
  end
  T = U * S;

  a_cand = T * randn(size(T, 2), N_target);
  a_cand = a_cand ./ sqrt(sum(a_cand.^2, 1));

  %% random loc in [-1,1]^n
  b_cand = (2 * rand(1, N_target) - 1) * radius;

  %figure(3000);
  %plot_hyper(a_cand, b_cand);

end
