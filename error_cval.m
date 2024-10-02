function [err_train, err_test] = error_cval(p, u, y_d, y_test)

omegas = u.x;
coeff = u.u';

Khat = p.k(p, p.xhat, omegas);
Ktest = p.k(p, p.xhat_test, omegas);

F_train = p.obj.F(Khat * coeff - y_d);
F_test = p.obj.F(Ktest * coeff - y_test);

F_zero = p.obj.F(-y_d) + p.obj.F(-y_test);

err_train = sqrt(2*F_train) / sqrt(F_zero);
err_test = sqrt(F_train + F_test) / sqrt(F_zero);

end
