function  plot_kernel(p, weights, xc)

if nargin < 3
  xc = [0; 0];
end

if isstruct(weights)
  omegas = weights.x;
  Nomega = size(omegas, 2);
  coeff = abs(weights.u);
else
  omegas = weights;
  Nomega = size(omegas, 2);
  coeff = ones(1,Nomega);
end

wN = coeff / sum(coeff);

L = sqrt(2)*p.L;
  
x1 = linspace(-L, L, 50+1);
x2 = linspace(-L, L, 50+1);
[X1, X2] = meshgrid(x1, x2);

  
Khat = p.k(p, [X1(:)';X2(:)'], omegas);
Kc = p.k(p, xc, omegas);
kernel_c = Khat * (wN .* Kc)';

Y = reshape(kernel_c, length(x1), length(x2));

figure(4231)
surf(X1, X2, Y, 'EdgeColor', 'interp', 'FaceColor', 'none', 'LineWidth', 2);
maxval = max(Y(:));
minval = min(Y(:));
buffer = 0.1*(.1 + max(abs(maxval), abs(minval)));
axis([-L,L, -L,L, minval-buffer, maxval+buffer])
set(gca, 'FontSize', 12);

%figure(4230)
%surf(X1, X2, -log((X1-xc(1)).^2 + (X2-xc(2)).^2), 'EdgeColor', 'interp', 'FaceColor', 'none', 'LineWidth', 2);


end
