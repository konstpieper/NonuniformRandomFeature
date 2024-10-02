function [a_cand, b_cand, A, B] = sample_weights_nonlocal(xhat, f_d, deltaW, s, N_target)

  assert(s == 1 || s == 2)
  
  dim = size(xhat, 1);
  Nx = size(xhat, 2);

  [gf, Hf] = compute_derivatives(f_d, xhat);

  Agf = sum(gf, 2) / Nx;
  if norm(Agf, 2) == 0
    Agf = randn(dim, 1);
  end
  Agf = Agf / norm(Agf, 2);

  gf_norm = sqrt(sum(gf.^2, 1));

  if s == 1
    Tf = reshape(gf, dim, 1, Nx) .* reshape(gf, 1, dim, Nx);
  elseif s == 2
    Tf = zeros(dim, dim, Nx);
    for d = 1:dim
      Tf = Tf + Hf(:,d,:) .* Hf(d,:,:);
    end
  end

  %% distance dependent kernel
  Kxx = xhat' * xhat;
  D2 = diag(Kxx) + diag(Kxx)' - 2*Kxx;
  %%Kdist = exp(- (1/2) * D2 / (2*deltaW)^2);
  Kdist = exp(- sqrt(D2) / (2*deltaW));

  %% tensorproduct Tf * Kdist
  %KTf = sum(Tf .* reshape(Kdist, [1, 1, size(Kdist)]), 3);
  KTf = reshape(Tf, [dim*dim, Nx]) * Kdist.^2;

  Id = eye(dim);
  trKTf = sum(Id(:) .* KTf, 1);
  KTf = reshape(KTf, [dim, dim, Nx]);
  
  Lf = sqrt(trKTf) + eps;
  Lf = Lf / sum(Lf);
  cumLf = [0, cumsum(Lf)];
  cumLf = cumLf / cumLf(end);
  cumLf(end) = (1+eps);

  a_select = zeros(dim, 0);
  b_select = zeros(1, 0);
  N_select = 0;
  while N_select < N_target;

    %% number of planes to sample in one iteration
    Nit = N_target;%10;
    
    %% sample idx according to Lf = cumLf'
    [count, idx] = histc(rand(Nit, 1), cumLf);
    idx = sort(idx);

    %%TODO would need sqrtm or cholesky here
    %%a_hat = reshape(sum(KTf(:,:,idx) .* randn([dim, 1, Nit]), 1), [dim, Nit]);
    if s == 1
      W = randn([1, Nit, Nx]);
      gfW = reshape(gf, [dim, 1, Nx]) .* W;
      a_hat = reshape(sum(gfW .* reshape(Kdist(idx,:), [1, Nit, Nx]), 3), [dim, Nit]);
    elseif s == 2
      W = randn([1, dim, Nit, Nx]);
      HfW = reshape(Hf, [dim, dim, 1, Nx]) .* W;
      a_hat = reshape(sum(sum(HfW .* reshape(Kdist(idx,:), [1, 1, Nit, Nx]), 2), 4), [dim, Nit]);
      %HfW = reshape(sum(reshape(Hf, [dim, dim, 1, Nx]) .* W, 2), [dim, 1, Nx]);
      %a_hat = reshape(sum(HfW .* reshape(Kdist(idx,:), [1, Nit, Nx]), 3), [dim, Nit]);
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
