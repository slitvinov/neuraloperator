s = 256;
alpha = 2;
tau = 3;
norm_a = GRF(alpha, tau, s);
lognorm_a = exp(norm_a);

thresh_a = zeros(s, s);
thresh_a(norm_a >= 0) = 12;
thresh_a(norm_a < 0) = 4;
f = ones(s, s);

lognorm_p = solve_gwf(lognorm_a, f);
thresh_p = solve_gwf(thresh_a, f);

%Plot coefficients and solutions
[X, Y] = meshgrid(linspace(0, 1, s));
fig = figure('visible', 'off');
subplot(2, 2, 1);
pcolor(lognorm_a);
view(2);
shading interp;
colorbar;
subplot(2,2,2)
surf(X,Y,lognorm_p); 
view(2); 
shading interp;
colorbar;
subplot(2,2,3)
surf(X,Y,thresh_a); 
view(2); 
shading interp;
colorbar;
subplot(2,2,4)
surf(X,Y,thresh_p); 
view(2); 
shading interp;
colorbar;
saveas(fig, 'run.png')
