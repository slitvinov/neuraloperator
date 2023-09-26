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
[X, Y] = meshgrid(linspace(0, 1, s));
pathes = {'lognorm_a.png', 'lognorm_p.png', 'thresh_a.png', ...
	  'thresh_p.png'};
fields = {lognorm_a, lognorm_p, thresh_a, thresh_p};
fig = figure('visible', 'off');
for k = 1:4
  clf(fig);
  pcolor(fields{k});
  shading interp;
  axis('equal');
  saveas(fig, pathes{k});
end
