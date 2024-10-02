function bpts = find_bpoints(q, dq, sigma)

%% find stepsizes tau such that
%%  |q_tau| = |q + \tau dq| = sigma for at least one entry of q_tau
  
set_m = (q < -sigma) & (dq > 0);
set_0 = (q >= -sigma) & (q <= sigma);
set_p = (q > sigma) & (dq < 0);

taus = [ (               -sigma - q(set_m)) ./ dq(set_m); ...
	 (                sigma - q(set_p)) ./ dq(set_p); ...
	 (sign(dq(set_0))*sigma - q(set_0)) ./ dq(set_0)];

bpts = sort([taus(taus > 0 & taus < 1); 1]);

end
