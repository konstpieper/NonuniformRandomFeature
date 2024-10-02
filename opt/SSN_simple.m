function u = SSN_simple(p, Sred, ref, alpha, phi, u0)
%{ 
%% Solve the problem: 
            min_u 1/(2*K)*|Sred*u - ref|^2 + alpha*|u|_1
with a (damped) semismooth Newton method

Discription:

Input: 
  p - the model
  Sred - current value of the network and the relues
  ref - output data
  alpha - regularization parameter
  u0 - current coefficients
  
Output:
  u -  optimal variable

%}

[K, N] = size(Sred);

% nonnegative constraint
nonnegative = (isfield(p, 'nonnegative') && p.nonnegative);

% some operators
obj = @(u) sumsq(Sred*u - ref) / (2*K) + alpha * sum(phi.phi(abs(u)));

% ssn constant
c = 1 + alpha*phi.gamma;

% nonconvex part (only for scalar sparsity)
Dphima = @(u)  (phi.dphi(abs(u)) - 1) .* sign(u);
DDphima = @(u) phi.ddphi(abs(u));

% Prox and Robinson normal map residual
if ~nonnegative
  Pc = @(q) sign(q) .* max(0, abs(q) - alpha/c);
  DPc = @(q) abs(q) > alpha/c;
else
  Pc = @(q) max(0, q - alpha/c);
  DPc = @(q) q > alpha/c;
end

G = @(q, u) c*(q - u) + alpha*Dphima(u) + Sred'*(Sred*u - ref) / K;

%% initial q
gf0 = Sred'*(Sred*u0 - ref) / K;
nu0 = abs(u0);
gf0(nu0 > 0) = - alpha * sign( u0(nu0 > 0) );
ng0 = abs(gf0);
gf0(nu0 == 0 & ng0 > alpha) = (1 - 1e-14) * alpha * sign( gf0(nu0 == 0 & ng0 > alpha) );
q = u0 - 1/c*gf0;

% it should hold Pc(q) == u0
%disp(norm(Pc(q) - u0));

%% initialize SSN
u = Pc(q);
j = obj(u);
Gq = G(q, u);

j0 = j;
normGQ0 = norm(Gq);

iter = 0;
iterls = 0;
converged = false;

theta_old = 1e16;

reduced_solve = true;

while ~converged && iter < 6666 && iterls < 30000

    if reduced_solve
      II = DPc(q);
      HI = Sred(:,II)'*Sred(:,II) / K;
      epsi = 1e-12*norm(HI, inf);

      %% Semismooth Newton update
      dq = zeros(size(q));
      dq(II) = - (HI + epsi*eye(sum(II))) \ Gq(II);
      dq(~II) = - (Gq(~II) + Sred(:,~II)'*(Sred(:,II)*dq(II)) / K) / c;
    else
      DPc = diag(DPc(q));
      DG = c*(eye(N) - DPc) + Sred'*Sred*DPc / K;
      epsi = 1e-12*norm(DG, inf);

      %% Semismooth Newton update
      dq = - (DG + epsi*DPc) \ Gq;
    end

    qnew = q + dq;
    unew = Pc(qnew);
    jnew = obj(unew);
    tau = 1;

    taus = find_bpoints(q, dq, alpha/c);

    %% Possible damping
    theta = min(theta_old, tau);
    while isnan(jnew) || jnew > (1 + 1000*eps)*j
        % require descent only up to 10 times machine precision
        % for (jnew > j), the line-search does not terminate, occasionally
 
        %% damped Newton update
        tau = min([ max(taus(taus < theta)), theta]);
        qnew = q + (1 + 1e-14) * tau * dq;

        unew = Pc(qnew);
        jnew = obj(unew);
        theta = theta / 2;
        iterls = iterls + 1;

	%keyboard;
    end
    theta_old = theta * 10;

    q = qnew;
    u = unew;
    jold = j;
    j = jnew;

    Gq = G(q, u);

    fprintf('\t\tssn: %i, j=%1.2e, supp: %i, desc: %1.1e, res: %1.1e, damp: %1.1e\n', ...
    	     iter, j, nnz(abs(u)), jold-j, norm(Gq), tau);
    iter = iter + 1;

    converged = ((tau == 1) && (norm(Gq) < max(1e-8, 1e-10*normGQ0)));

end
fprintf('\tssn: %i, j=%1.2e, supp: %i, desc: %1.1e, res: %1.1e, damp: %1.1e\n', ...
         iter, j, nnz(abs(u)), j0-j, norm(Gq), tau);

%fprintf('\tssn iter %i, damping: %i\n', iter, iterls);

end
