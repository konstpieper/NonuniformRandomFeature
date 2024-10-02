function y = robot_arm(x)

  dim = size(x, 1);
  
  if mod(dim, 2) == 0
    Ls = x(1:2:end, :);
    thetas = x(2:2:end, :);
  else 
    %% in odd dimensions, add the first arm angle as zero (no influence on length)
    Ls = x(1:2:end, :);
    thetas = zeros(size(Ls));
    thetas(2:end, :) = x(2:2:end, :);
  end

  CStheta = cumsum(thetas, 1);

  u = sum(Ls .* cos(CStheta), 1);
  v = sum(Ls .* sin(CStheta), 1);

  y = sqrt(u.^2 + v.^2);

end
