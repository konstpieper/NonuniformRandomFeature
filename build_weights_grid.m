function [a_cand, b_cand] = build_weights_grid(xhat)

  xhat_norm = sqrt(sum(xhat.^2, 1));

  a_cand = xhat ./ xhat_norm;
  a_rand = rand(size(xhat,1), sum(xhat_norm == 0));
  a_rand = a_rand ./ sqrt(sum(a_rand.^2, 1));
  a_cand(:, xhat_norm == 0) = a_rand;

  a_cand = [a_cand, -a_cand];

  b_cand = [-xhat_norm, xhat_norm];

  a_cand = [a_cand, zeros(size(xhat,1), 1)];
  b_cand = [b_cand, 1];

  %figure(1234)
  %ts = [-1,1];
  %a_perp = [a_cand]

end
