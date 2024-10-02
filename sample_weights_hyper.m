function [a_cand, b_cand, A, B] = sample_weights_hyper(p, f_d, N_target)

  s = p.s;
  assert(s == 1 || s == 2)
  
  dim = size(p.xhat, 1);
  Nx = size(p.xhat, 2);

  [gf, Hf] = compute_derivatives(f_d, p.xhat);

  Agf = sum(gf, 2) / Nx;
  if norm(Agf, 2) == 0
    Agf = randn(dim, 1);
  end
  Agf = Agf / norm(Agf, 2);

  Na = 10*Nx;
  [a, b] = sample_weights_uniform(p.dim, p.R, Na);
  omegas = stereo(a, b);
  
  cauchy = @(al, x) 1 ./ (1 - 1i*x).^(al+1);
  if mod(s, 2) == 1
    eta_d = @(x) real( cauchy(p.dim+s-2, 1/sqrt(p.dim+s-1) * x) );
  else
    eta_d = @(x) imag( cauchy(p.dim+s-2, 1/sqrt(p.dim+s-1) * x) );
  end

  eta_x = p.k_oc(p, p.xhat, omegas, eta_d);

  g_tilde = sum(gf .* reshape(eta_x, [1, size(eta_x)]), 2);
  g_bar = 1;

  a_avg = reshape(g_tilde ./ g_bar, size(a));
  La = sum(a_avg .* a, 1);

  a_tilde = a;
  b_tilde = b;
  
  Lf = abs(La);
  maxLf = max(Lf);

  a_select = zeros(dim, 0);
  b_select = zeros(1, 0);
  N_select = 0;
  while N_select < N_target;

    sgn_perturb = sign(randn(1, Na));
 
    b_qual = maxLf*rand(1, Na);

    [delta_select, hat_select] = find(b_qual <= Lf);
    dh_select = sub2ind(size(b_qual), delta_select, hat_select);

    N_select = N_select + length(delta_select);
    a_select = [a_select, sgn_perturb(dh_select) .* a_tilde(:, hat_select)];
    b_select = [b_select, sgn_perturb(dh_select) .* b_tilde(dh_select)];

  end
  perm = randperm(N_select);
  a_select = a_select(:, perm(1:N_target));
  b_select = b_select(:, perm(1:N_target));
    
  a_cand = a_select;
  b_cand = b_select;

  if s == 1
    A = [zeros(dim, 1)];
    B = [1];
  else
    A = [Agf, -Agf, zeros(dim, 1)];
    B = [0, 0, 1];
  end

  %figure(3000);
  %plot_hyper(a_cand, b_cand);

end
