function [a_cand, b_cand] = sample_weights_uniform(dim, radius, N_target)

  %% random unit vectors
  a_cand = randn(dim, N_target);
  a_cand = a_cand ./ sqrt(sum(a_cand.^2, 1));

  %% random loc in [-R, R]
  b_cand = (2 * rand(1, N_target) - 1) * radius;

  %figure(3000);
  %plot_hyper(a_cand, b_cand);
  
end
