rand('state', 123456);
s = 421;
alpha = 2;
tau = 3;
norm_a = GRF(alpha, tau, s);
a = exp(norm_a);
f = ones(s, s);
u = solve_gwf(a, f);
[X, Y] = meshgrid(linspace(0, 1, s));
pathes = {'a.png', 'u.png'};
fields = {a, u};
fig = figure('visible', 'off');
for k = 1:length(pathes)
  clf(fig);
  colormap('jet');
  axis('equal');
  pcolor(fields{k});
  shading('flat');  
  colorbar();
  print("-S1200,900", pathes{k})
end
clf(fig);
contour(u, 'linecolor', 'black');
axis('equal');
print("-S1200,900", "contour.png")
