function run()
  N = 1;
  gamma = 2.5;
  tau = 7;
  sigma = 7^(2);
  visc = 1/1000;
  s = 1024;
  steps = 200;
  input = zeros(N, s);
  if steps == 1
    output = zeros(N, s);
  else
    output = zeros(N, steps, s);
  end
  tspan = linspace(0,1,steps+1);
  x = linspace(0,1,s+1);
  for j=1:N
    u0 = GRF1(s/2, 0, gamma, tau, sigma);
    u = burgers1(u0, tspan, s, visc);
    u0eval = u0(x);
    input(j,:) = u0eval(1:end-1);
    if steps == 1
      output(j,:) = u.values;
    else
      for k=2:(steps+1)
	output(j,k,:) = u{k}.values;
      end
    end
    disp(j);
  end
endfunction
function u = GRF1(N, m, gamma, tau, sigma)
  my_const = 2*pi;
  my_eigs = sqrt(2)*(abs(sigma).*((my_const.*(1:N)').^2 + tau^2).^(-gamma/2));
  xi_alpha = randn(N,1);
  alpha = my_eigs.*xi_alpha;
  xi_beta = randn(N,1);
  beta = my_eigs.*xi_beta;
  a = alpha/2;
  b = -beta/2;
  c = [flipud(a) - flipud(b).*1i;m + 0*1i;a + b.*1i];
  uu = chebfun(c, [0 1], 'trig', 'coeffs');
  u = chebfun(@(t) uu(t - 0.5), [0 1], 'trig');
endfunction
function u = burgers1(init, tspan, s, visc)
  S = spinop([0 1], tspan);
  dt = tspan(2) - tspan(1);
  S.lin = @(u) + visc*diff(u,2);
  S.nonlin = @(u) - 0.5*diff(u.^2);
  S.init = init;
  u = spin(S,s,dt,'plot','off');
endfunction
