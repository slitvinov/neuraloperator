function P = solve_gwf(c, F)
  K = length(c);
  [X1,Y1] = meshgrid(1/(2*K):1/K:(2*K-1)/(2*K),
		     1/(2*K):1/K:(2*K-1)/(2*K));
  [X2,Y2] = meshgrid(0:1/(K-1):1,0:1/(K-1):1);
  c = interp2(X1, Y1, c, X2, Y2,'spline');
  F = interp2(X1, Y1, F, X2, Y2,'spline');
  F = F(2:K-1,2:K-1);
  d = cell(K-2,K-2);
  [d{:}] = deal(sparse(zeros(K-2)));
  for j=2:K-1
    d{j-1,j-1} = spdiags([[-(c(2:K-2,j)+c(3:K-1,j))/2;0],...
			  (c(1:K-2,j)+c(2:K-1,j))/2 ...
			  + (c(3:K,j)+c(2:K-1,j))/2 ...
			  + (c(2:K-1,j-1)+c(2:K-1,j))/2 ...
			  + (c(2:K-1,j+1)+c(2:K-1,j))/2,...
			  [0;-(c(2:K-2,j)+c(3:K-1,j))/2]],...
			 -1:1,K-2,K-2);
    if j ~= K-1
      d{j-1,j} = spdiags(-(c(2:K-1,j)+c(2:K-1,j+1))/2,0,K-2,K-2);
      d{j,j-1} = d{j-1,j};
    end
  end
  A = cell2mat(d)*(K-1)^2;
  P =[zeros(1,K);
      [zeros(K-2,1),vec2mat(A\F(:),K-2),zeros(K-2,1)];
      zeros(1,K)];
  P = interp2(X2, Y2, P, X1, Y1, 'spline')';
end
