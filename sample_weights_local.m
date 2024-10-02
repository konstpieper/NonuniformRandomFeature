function [a_cand, b_cand, A, B] = sample_weights_local(xhat, f_d, deltaW, s, N_target)

  assert(s == 1 || s == 2)
  
  dim = size(xhat, 1);
  Nx = size(xhat, 2);
  
  [gf, Hf] = compute_derivatives(f_d, xhat);

  Agf = sum(gf, 2) / Nx;
  if norm(Agf, 2) == 0
    Agf = randn(dim, 1);
  end
  Agf = Agf / norm(Agf, 2);

  if s == 1
    Lf = sqrt(sumsq(gf, 1)) + eps;
  elseif s == 2
    %Lf = abs(sum(sum(Hf .* eye(dim))));
    Lf = sqrt(sum(sumsq(Hf, 1), 2)) + eps;
    Lf = reshape(Lf, 1, Nx);
  end
  Lf = Lf / sum(Lf);
  cumLf = [0, cumsum(Lf)];
  cumLf = cumLf / cumLf(end);
  cumLf(end) = (1+eps);

  a_select = zeros(dim, 0);
  b_select = zeros(1, 0);
  N_select = 0;
  while N_select < N_target;

    %% number of hyperplanes to sample in one iteration
    Nit = N_target;%10;
    
    %% sample idx according to Lf = cumLf'
    [count, idx] = histc(rand(Nit, 1), cumLf);
    idx = sort(idx);
    
    if s == 1
      a_hat = gf(:,idx) .* randn([1, Nit]);
    elseif s == 2
      a_hat = reshape(sum(Hf(:,:,idx) .* randn([dim, 1, Nit]), 1), [dim, Nit]);
    end

    ah_norm = sqrt(sum(a_hat.^2, 1));
    a_hat = a_hat ./ ah_norm;

    b_hat = - sum(xhat(:,idx) .* a_hat, 1);
    b_hat = b_hat + deltaW * randn(1, Nit);

    N_select = N_select + Nit;
    a_select = [a_select, a_hat];
    b_select = [b_select, b_hat];

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
