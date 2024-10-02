function hl = plot_with_error(x, y, style, colour, leg)

mode = 'log';

if strcmp(mode, 'log')
  ly = log(max(y, 1e-16));
  lym = mean(ly, 1);
  le = std(ly, [], 1);  
  
  x = x(:);
  ym = exp(lym(:));
  yplus = exp(lym(:) + le(:));
  yminus = exp(lym(:) - le(:));
else
  ym = mean(y, 1);
  e = std(y, [], 1);  
  
  x = x(:);
  ym = ym(:);
  yplus = ym(:) + e(:);
  yminus = abs(ym(:) - e(:));
end

newplot
hl = line(x, ym);
hp = patch([x; x(end:-1:1); x(1)], [yminus; yplus(end:-1:1); yminus(1)], 'b');

switch colour
  case 'blue'
    col = "#0072BD";
  case 'orange'
    col = "#D95319";
  case 'yellow'
    col = "#EDB120";
  case 'purple'
    col = "#7E2F8E";
  case 'green'
    col = "#77AC30";
## [0.3010 0.7450 0.9330]	"#4DBEEE"	
## Sample of RGB triplet [0.3010 0.7450 0.9330], which appears as light blue
## [0.6350 0.0780 0.1840]	"#A2142F"	
## Sample of RGB triplet [0.6350 0.0780 0.1840], which appears as dark red
  otherwise
    col = "#000000";
end

%legend('show')
set(hl, 'color', col, 'linestyle', style, 'linewidth', 1, 'displayname', leg);
%legend('autoupdate', 'off');
set(hp, 'facecolor', col, 'facealpha', 0.1, 'edgecolor', 'none');

if strcmp(mode, 'log')
  set(gca, 'XScale', 'log', 'Yscale', 'log')
end

end
