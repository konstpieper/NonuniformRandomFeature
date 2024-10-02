function [om, delta] = stereo(a, b)

%% om = stereo(a',b') is the stereographic projection
%% of the normalized vectors (a',b') = (a,b) / sqrt(|a|^2 + b^2)
%% from the point S = (a_1,b_1) = (0,-1)

%% om = a' / (1 + b');
%%    = a  / (sqrt(|a|^2 + b^2) + b);

a2 = sum(a.^2, 1);
om = a ./ (sqrt(a2 + b.^2) + b);

%% delta is the inverse of the norm of a 
delta = 1 ./ sqrt(a2);

end
