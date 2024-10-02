function plot_hyper(A, B, C)
  
  ts = [-3; 3];
  A_norm_2 = A(1,:).^2 + A(2,:).^2;
  A_perp = [A(2,:); -A(1,:)] ./ sqrt(A_norm_2);
  x1_pl = - A(1,:) .* B ./ A_norm_2 + ts .* A_perp(1,:);
  x2_pl = - A(2,:) .* B ./ A_norm_2 + ts .* A_perp(2,:);
  plot(x1_pl, x2_pl, 'k');
  if nargin > 2
    plot(x1_pl, x2_pl, 'k', 'linewidth', .05);
    hold on;
    lw = abs(C) / sum(abs(C)) * 20;
    for k = find(lw > 0);
      %disp([k, lw(k)])
      plot(x1_pl(:,k), x2_pl(:,k), 'k', 'linewidth', lw(k));
    end
    hold off;
  end
  axis([-1,1,-1,1] / sqrt(2));
				%drawnow;

end
