function [k, dk, ddk] = sigmoid(y, S);

%% For S = 2:
%%   kernel = sp(y) = log(1 + exp(y))
%% For S < 2:
%%   kernel = (d/dy)^{2-S} sp(y) * (2-S)!

my = max(y, 0);
emmy = exp(- my);
ey = exp(y - my);
if S == 0
  k     = emmy .* ey ./ (emmy + ey).^2;
  if nargout > 1
    dk = k .* (emmy - ey) ./ (emmy + ey);
  end
elseif S == 1
  k     = ey ./ (emmy + ey);
  if nargout > 1
    dk  = emmy .* ey ./ (emmy + ey).^2;
  end
  if nargout > 2
    ddk = dk .* (emmy - ey) ./ (emmy + ey);
  end
elseif S == 2
  k     = my + log(emmy + ey);
  if nargout > 1
    dk  = ey ./ (emmy + ey);
  end
  if nargout > 2
    ddk = emmy .* ey ./ (emmy + ey).^2;
  end
else
  error("exponent s=%d not implemented", S)
end

end
