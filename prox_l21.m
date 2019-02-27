function z = prox_l21(t, lambda, x)

l2x_row_wise = sqrt(sum(x.^2, 2)); % row-wise l2 norm
term = 1 - 1 ./ max(l2x_row_wise ./ t .* lambda, 1);
z = bsxfun(@times, x, term);

end

%function z = prox_l21(t, lambda, x)
%    z = zeros(size(x));
%    for i = 1 : size(x, 1)
%        xi = x(i, :);
%        l2x = norm(x, 2);
%        if l2x == 0
%            z(i, :) = zeros(size(xi));
%        else
%            z(i, :) = max(l2x - t * lambda, 0) / l2x * xi;
%        end
%    end
%end
