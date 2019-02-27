function res = l21_log_metro_admm(X, Y, lambdas, is_test)

lambda1 = lambdas(1); % lambdas
lambda2 = lambdas(2); % lambdas
lambda3 = lambdas(3); % lambdas
lambda4 = lambdas(4); % lambdas
lambda5 = lambdas(5); % lambdas
rho = 1; % rho
n = size(X{1}, 2); % num of features
ts = length(X); % num of tasks
W = zeros([n, ts]); % initialize primal variables 
Uw = zeros([n, ts]); % initialize dual variables
U1 = zeros([n, ts]); % initialize lagrangian multipliers
% One way to guess W ===========================================================
% W0 = [];
% for j = 1: ts
%    W0 = cat(2, W0, double(X{j}') * double(Y{j}));
% end
% W = W0;
% End of one way to guess W ====================================================
% stop condition
er = 1e-3; % stop condition 1: primial residual
es = 1e-3; % stop condition 2: dual residual
MAX_ITER = 1000; % max iters for main problem
GRAD_MAX_ITER = 1000; % max iters for subproblems
COR_MAX_ITER = 1000; % max iters for subproblems
GRAD_TOL = 1e-1; % stop condition for subproblems
COR_TOL = 1;
con1_delta = {};
con2_delta = {};
con3_delta = {};
con4_delta = {};

for iter = 1 : MAX_ITER
    W = update_W(); % update W 
    %fun_loss(W)
    %regularizer = lambda1 * norm(W)^2 + lambda2 * norm(W(:,3) - W(:,5))^2 + ...
    %    + lambda3 * norm(W(:,1) - W(:,5))^2 ...
    %    + lambda4 * norm(W(:,6) - W(:,2))^2 ...
    %    + lambda5 * norm(W(:,1) - W(:,6))^2
    Uw_new = prox_l21(1 / rho, lambda1, U1 + W); % update Uw
    U1 = U1 + rho * (W - Uw_new); % update U1
    r = norm(W - Uw_new); % update primal residual
    s = rho * (norm(Uw_new - Uw)); % update dual residual
    Uw = Uw_new; % update dual variable
    % Stop condition
    %if is_test
    %    disp(iter); disp(r); disp(s);
    %end
    if iter > 1 && r < er && s < es
        break;
    end
    % Speeding ADMM ============================================================
%     if r > 10 * s % Update rho
%         rho = 2 * rho;
%     elseif 10 * r < s
%         rho = rho / 2;
%     end
    
    con1_delta{iter} = norm(norm(W(:, 3) - W(:, 5))^2);
    con2_delta{iter} = norm(norm(W(:, 1) - W(:, 5))^2);
    con3_delta{iter} = norm(norm(W(:, 2) - W(:, 6))^2);
    con4_delta{iter} = norm(norm(W(:, 1) - W(:, 6))^2);
    % End of speeding ADMM ======================================================
end

res.W = W; 
res.Uw = Uw; 
res.U1 = U1; 
res.lambdas = lambdas; 
res.con1_delta = con1_delta;
res.con2_delta = con2_delta;
res.con3_delta = con3_delta;
res.con4_delta = con4_delta;

% Update W method ==============================================================
    function W_new = update_W()
        W_new = W;
        fun_vals = []; fun_vals = cat(1, fun_vals, fun_val(W_new));
        for c_iter = 1 : COR_MAX_ITER 
            W_new(:,1) = update_wj(W_new, 1);
            W_new(:,2) = update_wj(W_new, 2);
            W_new(:,3) = update_wj(W_new, 3);
            W_new(:,4) = update_wj(W_new, 4);
            W_new(:,5) = update_wj(W_new, 5);
            W_new(:,6) = update_wj(W_new, 6);
            fun_vals = cat(1, fun_vals, fun_val(W_new));
            if c_iter > 1 ...
                && abs(fun_vals(c_iter) - fun_vals(c_iter - 1)) < COR_TOL 
                break;
            end
        end
    end

    % Use gradient descent to update W
    function Wj_new = update_wj(W_new, j)
        wj_grad_step = 1e-6; wjs = []; wjs = cat(1, wjs, W_new(:,j)');
        for k_wj = 1 : GRAD_MAX_ITER
            W_new_tmp = W_new;
            while true
                grad_wj_val = grad_wj(W_new, j);
                W_new_tmp(:,j) = W_new(:,j) - wj_grad_step * grad_wj_val;
                left = fun_val(W_new_tmp);
                right = fun_val(W_new) - ...
                    .5 * wj_grad_step * grad_wj_val' * grad_wj_val;
                if  left <= right 
                    break;
                else
                    wj_grad_step = .5 * wj_grad_step;
                end
            end
            W_new = W_new_tmp;
            wjs = cat(1, wjs, W_new(:,j)');
            %if k_wj > 1
            %    norm(wjs(k_wj, :) - wjs(k_wj - 1, :))
            %end
            if k_wj > 1 && norm(wjs(k_wj,:) - wjs(k_wj - 1,:)) < GRAD_TOL 
                break;
            end
        end
        Wj_new = W_new(:,j); 
    end

    % Compute grad wj
    function var = grad_wj(cur_w, j)
        zj = -Y{j} .* (X{j} * cur_w(:,j));
        var = 1 / ts * X{j}' * (-Y{j} .* (1 - 1 ./ (1 + exp(zj)))) + U1(:,j) ...
            + rho * (cur_w(:,j) - Uw(:,j));
        if j == 1
            var = var + 2 * lambda3 * (cur_w(:,1) - cur_w(:,5)) ...
                + 2 * lambda5 * (cur_w(:,1) - cur_w(:,6));
        elseif j == 2
            var = var + 2 * lambda4 * (cur_w(:,2) - cur_w(:,6));
        elseif j == 3
            var = var + 2 * lambda2 * (cur_w(:,3) - cur_w(:,5));
        elseif j == 5
            var = var - 2 * lambda2 * (cur_w(:,3) - cur_w(:,5)) ...
                - 2 * lambda3 * (cur_w(:,1) - cur_w(:,5));
        elseif j == 6
            var = var + 2 * lambda4 * (cur_w(:,6) - cur_w(:,2)) ...
                - 2 * lambda5 * (cur_w(:,1) - cur_w(:,6));
        end
    end

    function loss = fun_loss(W_new)
        loss = 0;
        for j = 1 : ts
            loss = loss + sum(log(1 + exp(-Y{j} .* (X{j} * W_new(:,j)))));
        end
    end
    % Compute f
    function var = fun_val(W_new)
        var = 0;
        for j = 1 : ts
            var = var + sum(log(1 + exp(-Y{j} .* (X{j} * W_new(:,j))))) ...
                + sum(U1(:, j) .* (W_new(:, j) - Uw(:, j))) ...
                + rho / 2 * norm(W_new(:, j) - Uw(:, j))^2;
        end
        var = var / ts;
        var = var ...
            + lambda2 * norm(W_new(:,3) - W_new(:,5))^2 ...
            + lambda3 * norm(W_new(:,1) - W_new(:,5))^2 ...
            + lambda4 * norm(W_new(:,6) - W_new(:,2))^2 ...
            + lambda5 * norm(W_new(:,1) - W_new(:,6))^2;
    end
% End of update W method =======================================================
end
