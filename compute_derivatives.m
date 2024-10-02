function [g_d, H_d] = compute_derivatives(f_d, xhat);

  [dim, Nx] = size(xhat);
  g_d = zeros(dim, Nx);
  H_d = zeros(dim, dim, Nx);

  tau = sqrt(sqrt(eps));
  FD = @(f, dx) (f(xhat + tau*dx) - f(xhat - tau*dx)) / (2*tau);
  
  FD2 = @(f, dx1, dx2) (f(xhat + tau*dx1 + tau*dx2) - f(xhat + tau*dx1 - tau*dx2) ...
	              - f(xhat - tau*dx1 + tau*dx2) + f(xhat - tau*dx1 - tau*dx2)) / (4*tau^2);

  for d = 1:dim
    dx = ((1:dim) == d)';
    g_d(d,:) = FD(f_d, dx);
  end

  for d1 = 1:dim
    dx1 = ((1:dim) == d1)';
    for d2 = 1:dim
      dx2 = ((1:dim) == d2)';
      H_d(d1,d2,:) = FD2(f_d, dx1, dx2);
    end
  end

end
