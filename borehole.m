function y = borehole(x)

assert(size(x, 1) == 8);

%% transform from [-1,1] to [a,b]
transform = @(x, a, b) a + (b - a) .* (x + 1) / 2;

%rw ∈ [0.05,  0.15] 	radius of borehole (m)
%r  ∈ [100,   50000] 	radius of influence (m)
%Tu ∈ [63070, 115600]   transmissivity of upper aquifer (m2/yr)
%Hu ∈ [990,   1110] 	potentiometric head of upper aquifer (m)
%Tl ∈ [63.1,  116] 	transmissivity of lower aquifer (m2/yr)
%Hl ∈ [700,   820] 	potentiometric head of lower aquifer (m)
%L  ∈ [1120,  1680] 	length of borehole (m)
%Kw ∈ [9855,  12045] 	hydraulic conductivity of borehole (m/yr)

rw = transform(x(1,:), 0.05, 0.15);
r  = transform(x(2,:), 100, 50000);
Tu = transform(x(3,:), 63070, 115600);
Hu = transform(x(4,:), 990, 1110);
Tl = transform(x(5,:), 63.1, 116);
Hl = transform(x(6,:), 700, 820);
L  = transform(x(7,:), 1120, 1680);
Kw = transform(x(8,:), 9855, 12045);

%% evaluate borehole function
num = 2 * pi .* Tu .* (Hu - Hl);

lrrw = log(r ./ rw);

suma = 2 * L .* Tu / (lrrw .* rw.^2 .* Kw);
sumb = Tu ./ Tl;

y = num ./ lrrw .* (1 + suma + sumb);

end
