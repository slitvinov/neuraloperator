function y = idct (x, n)
  [nr, nc] = size (x);
  w = [ sqrt(4*n); sqrt(2*n)*exp((1i*pi/2/n)*[1:n-1]') ] * ones (1, nc);
  y = x.*w;
  w = exp(-1i*pi*[n-1:-1:1]'/n) * ones(1,nc);
  y = ifft ( [ y ; zeros(1,nc); y(n:-1:2,:).*w ] );
  y = y(1:n, :);
  y = real (y);
endfunction
