function [a, b] = unstereo(om, kappa)

%% (a',b') = unstereo(om) is the inverse stereographic projection
%% from the point S = (a_1,b_1) = (0,-1)
%%   b' = (1 - |om|^2) / (1 + |om|^2);
%%   a' =       2 * om / (1 + |om|^2);
  
%% the return value (a,b) will be normalized according to kappa
%%  (a,b) = (a',b') / sqrt(kappa*(b')^2 + |a'|^2)
%% which is
%%   b = (1 - |om|^2) / nor;
%%   a =       2 * om / nor;
%% where
%%   nor = sqrt(4*|om|^2 + kappa*(1 - |om|^2)^2)

om2 = sum(om.^2, 1);

if nargin < 2 || kappa == 1
  %% kappa = 1, normalization included in inverse stereo
  nor = (1 + om2);
  b = (1 - om2) ./ (1 + om2);
  a =    2 * om ./ (1 + om2);
else
  nor = sqrt(4 * om2 + kappa * (1 - om2).^2);
  b = (1 - om2) ./ nor;
  a =    2 * om ./ nor;  
end

end
