function U = GRF(alpha,tau,s)
  xi = randn(s, s);
  [K1, K2] = meshgrid(0:s-1,0:s-1);
  coef = tau^(alpha-1).*(pi^2*(K1.^2+K2.^2) + tau^2).^(-alpha/2);
  L = s*coef.*xi;
  L(1, 1) = 0;
  U = idct(idct(L, s).', s).';
end
