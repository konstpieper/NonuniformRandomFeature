function plot_density(p, A, B, C)

norm_a = sqrt(eps + sumsq(A, 1));
a = A ./ norm_a;
b = B ./ norm_a;
c = abs(C) / sum(abs(C));

the_h = linspace(-pi, pi, 60);
b_h = linspace(-1, 1, 30);
[T_h, B_h] = meshgrid(the_h, b_h);

dist_a = acos(a(1,:).*sin(T_h(:)) + a(2,:).*cos(T_h(:))) * 2/pi;
radi = 0.1;
rho_h = sum(c .* exp(-dist_a.^2/(2*radi^2) - (B_h(:) - b).^2/(2*radi^2)), 2);
rho_h = rho_h / sqrt((2*pi)^2 * radi^2 / 2 * pi);
%surf(T_h, B_h, reshape(rho_h, size(B_h)), 'EdgeColor', 'none', 'FaceColor', 'interp');
contour(T_h, B_h, reshape(rho_h, size(B_h)));
axis([-pi,pi, -1,1]);
view(0, 90);
grid off;

xlabel('$\angle a = \arccos(a_1)$', 'interpreter', p.label_interp)
ylabel('$b$', 'interpreter', p.label_interp)
set(gca, 'ticklabelinterpreter', p.label_interp)
drawnow;

end
