% UCM equation attempt using the collocation method with a
% Chebyshev polynomial basis, solving explicitly for f and g = f'. Boundary
% conditions are f(0) = g(0) = g(inf) - 1 = 0, along with Txx(inf) =
% Txy'(inf) = Tyy'(inf) = 0

tic

% Solution space
N = 99; N_cts = 1000;
nvals = 0 : 1 : N;
x_min = 0; x_max = 5;
x_vec_cts = x_min : (x_max - x_min) / N_cts : x_max;

% Split points of the solution vector
finit = 2; ffin = N + 1; ginit = 2; gfin = N; hinit = 1; hfin = N + 1;
Tauxxinit = 1; Tauxxfin = N; Tauxxpinit = 1; Tauxxpfin = N + 1;
Tauxyinit = 1; Tauxyfin = N + 1; Tauxypinit = 1; Tauxypfin = N;
Tauyyinit = 1; Tauyyfin = N + 1; Tauyypinit = 1; Tauyypfin = N;
fpts = ffin - finit + 1; gpts = gfin - ginit + 1; hpts = hfin - hinit + 1;
Tauxxpts = Tauxxfin - Tauxxinit + 1; Tauxxppts = Tauxxpfin - Tauxxpinit + 1;
Tauxypts = Tauxyfin - Tauxyinit + 1; Tauxyppts = Tauxypfin - Tauxypinit + 1;
Tauyypts = Tauyyfin - Tauyyinit + 1; Tauyyppts = Tauyypfin - Tauyypinit + 1;
x_spl = cumsum([fpts, gpts, hpts, Tauxxpts, Tauxxppts, Tauxypts, Tauxyppts, ...
    Tauyypts, Tauyyppts]);

% Parameters
Wi = 0.5;
beta = 0.7;

% x values
xvals = (1 / 2) * ((x_max + x_min) - (x_max - x_min) * cos(pi * nvals / N));

% x values in L [-1, 1] space
xLvals = (1 / (x_max - x_min)) * ((x_max + x_min) - 2 * xvals);
xLvals_cts = (1 / (x_max - x_min)) * ((x_max + x_min) - 2 * x_vec_cts);

% BCs
f0 = 0; fp0 = 0; fpN = 1; TxxN = 0; TxypN = 0; TyypN = 0;

% Initialise T and Tp matrices
Tmat = zeros(N + 1, N + 1);
Tpmat = Tmat;
Tmat(1, :) = ones(1, N + 1);
Tmat(2, :) = xLvals;

% Calculate T matrix
for i1 = 2 : N
    Tmat(i1 + 1, :) = 2 * xLvals .* Tmat(i1, :) - Tmat(i1 - 1, :);
end

% Calculate Tp matrix
for i1 = 1 : N
    for j1 = 0 : N
        for k1 = 0 : i1 - 1
            if (k1 == 0) || (k1 == N)
                c = 1 / 2;
            else
                c = 1;
            end

            if mod(k1+i1,2) == 1
                Tpmat(i1 + 1, j1 + 1) = Tpmat(i1 + 1, j1 + 1) + c * Tmat(k1 + 1, j1 + 1);
            end
        end
        
        Tpmat(i1 + 1, j1 + 1) = - 4 * i1 * Tpmat(i1 + 1, j1 + 1) / (x_max - x_min);
    end
end

% Calculate differentiation matrices
A0 = (2 / N) * ([(1 / 2) * Tpmat(1, :); Tpmat(2 : end - 1, :); (1 / 2) * Tpmat(end, :)]' * Tmat);
A0(:, 1) = (1 / 2) * A0(:, 1);
A0(:, end) = (1 / 2) * A0(:, end);

fprintf('Diff matrices defined.\n');

% Outsource the system to one vector valued function F to minimise
F = @(y) root_eqn_OB(y, N, x_spl, Wi, beta, A0, f0, fp0, fpN, TxxN, TxypN, TyypN);

% Set up the solver
algo_choice = 'levenberg-marquardt';
options = optimoptions('fsolve','MaxIter',1e+06,'MaxFunctionEvaluations',1e+07,...
    'FunctionTolerance',1e-18,'StepTolerance',1e-08,'Algorithm',algo_choice,...
    'Display','iter');

% Set initial f guess as the x values
Trialy = [xvals(finit : ffin)'; (1 / x_max) * xvals(ginit : gfin)'; ...
    (1 - (1 / x_max) * xvals(hinit : hfin))'; zeros(Tauxxpts, 1); ...
    zeros(Tauxxppts, 1); zeros(Tauxypts, 1); zeros(Tauxyppts, 1); - 2 ...
    * ones(Tauyypts, 1); zeros(Tauyyppts, 1)];
[y_sol, Err_coll_vec] = fsolve(F, Trialy, options);

% Separate f and g values from y_sol
f_sol_full = [f0; y_sol(1 : x_spl(1))];
g_sol_full = [fp0; y_sol(x_spl(1) + 1 : x_spl(2)); fpN];
h_sol_full = y_sol(x_spl(2) + 1 : x_spl(3));

% Re-produce BCs for Tau functions
TauxxN = TxxN;
TauxypN = TxypN - A0(end, :) * h_sol_full;
TauyypN = TyypN + 2 * h_sol_full(end);

% Separate Tij and tij values from y_sol
Tauxx_sol_full = [y_sol(x_spl(3) + 1 : x_spl(4)); TauxxN];
tauxx_sol_full = y_sol(x_spl(4) + 1 : x_spl(5));
Tauxy_sol_full = y_sol(x_spl(5) + 1 : x_spl(6));
tauxy_sol_full = [y_sol(x_spl(6) + 1 : x_spl(7)); TauxypN];
Tauyy_sol_full = y_sol(x_spl(7) + 1 : x_spl(8));
tauyy_sol_full = [y_sol(x_spl(8) + 1 : x_spl(9)); TauyypN];

% Post-processing on xvals space
a_f_consts = (2 / N) * Tmat * [(1 / 2) * f_sol_full(1); f_sol_full(2 : end - 1); ...
    (1 / 2) * f_sol_full(end)];
a_g_consts = (2 / N) * Tmat * [(1 / 2) * g_sol_full(1); g_sol_full(2 : end - 1); ...
    (1 / 2) * g_sol_full(end)];
a_h_consts = (2 / N) * Tmat * [(1 / 2) * h_sol_full(1); h_sol_full(2 : end - 1); ...
    (1 / 2) * h_sol_full(end)];
a_Tauxx_consts = (2 / N) * Tmat * [(1 / 2) * Tauxx_sol_full(1); Tauxx_sol_full(2 : end - 1); ...
    (1 / 2) * Tauxx_sol_full(end)];
a_tauxx_consts = (2 / N) * Tmat * [(1 / 2) * tauxx_sol_full(1); tauxx_sol_full(2 : end - 1); ...
    (1 / 2) * tauxx_sol_full(end)];
a_Tauxy_consts = (2 / N) * Tmat * [(1 / 2) * Tauxy_sol_full(1); Tauxy_sol_full(2 : end - 1); ...
    (1 / 2) * Tauxy_sol_full(end)];
a_tauxy_consts = (2 / N) * Tmat * [(1 / 2) * tauxy_sol_full(1); tauxy_sol_full(2 : end - 1); ...
    (1 / 2) * tauxy_sol_full(end)];
a_Tauyy_consts = (2 / N) * Tmat * [(1 / 2) * Tauyy_sol_full(1); Tauyy_sol_full(2 : end - 1); ...
    (1 / 2) * Tauyy_sol_full(end)];
a_tauyy_consts = (2 / N) * Tmat * [(1 / 2) * tauyy_sol_full(1); tauyy_sol_full(2 : end - 1); ...
    (1 / 2) * tauyy_sol_full(end)];

% Initialise T_cts, Tp_cts and Tpp_cts matrices
T_cts_mat = zeros(N + 1, N_cts + 1);
Tp_cts_mat = T_cts_mat; Tpp_cts_mat = T_cts_mat; Tppp_cts_mat = T_cts_mat;
T_cts_mat(1, :) = ones(1, N_cts + 1);
T_cts_mat(2, :) = xLvals_cts;

% Calculate T_cts matrix
for i1 = 2 : N
    T_cts_mat(i1 + 1, :) = 2 * xLvals_cts .* T_cts_mat(i1, :) - T_cts_mat(i1 - 1, :);
end

% Calculate Tp_cts matrix
for i1 = 1 : N
    for j1 = 0 : N_cts
        for k1 = 0 : i1 - 1
            if (k1 == 0) || (k1 == N)
                c = 1 / 2;
            else
                c = 1;
            end

            if mod(k1+i1,2) == 1
                Tp_cts_mat(i1 + 1, j1 + 1) = Tp_cts_mat(i1 + 1, j1 + 1) + c ...
                    * T_cts_mat(k1 + 1, j1 + 1);
            end
        end
        
        Tp_cts_mat(i1 + 1, j1 + 1) = - 4 * i1 * Tp_cts_mat(i1 + 1, j1 + 1) / (x_max - x_min);
    end
end

% Calculate Tpp_cts matrix
for i1 = 2 : N
    for j1 = 0 : N_cts
        for k1 = 0 : i1 - 1
            if (k1 == 0) || (k1 == N)
                c = 1 / 2;
            else
                c = 1;
            end

            if mod(k1+i1,2) == 1
                Tpp_cts_mat(i1 + 1, j1 + 1) = Tpp_cts_mat(i1 + 1, j1 + 1) + c ...
                    * Tp_cts_mat(k1 + 1, j1 + 1);
            end
        end
        
        Tpp_cts_mat(i1 + 1, j1 + 1) = - 4 * i1 * Tpp_cts_mat(i1 + 1, j1 + 1) / (x_max - x_min);
    end
end

% Calculate Tppp_cts matrix
for i1 = 3 : N
    for j1 = 0 : N_cts
        for k1 = 0 : i1 - 1
            if (k1 == 0) || (k1 == N)
                c = 1 / 2;
            else
                c = 1;
            end

            if mod(k1+i1,2) == 1
                Tppp_cts_mat(i1 + 1, j1 + 1) = Tppp_cts_mat(i1 + 1, j1 + 1) + c ...
                    * Tpp_cts_mat(k1 + 1, j1 + 1);
            end
        end
        
        Tppp_cts_mat(i1 + 1, j1 + 1) = - 4 * i1 * Tppp_cts_mat(i1 + 1, j1 + 1) / (x_max - x_min);
    end
end

% Initialise doubles vectors
f_vals = zeros(1, N_cts + 1); d1f_vals = f_vals; d2f_vals = f_vals;
d3f_vals = f_vals;
g_vals = f_vals; d1g_vals = f_vals; d2g_vals = f_vals;
h_vals = f_vals; d1h_vals = f_vals;
Tauxx_vals = f_vals; d1Tauxx_vals = f_vals;
tauxx_vals = f_vals;
Tauxy_vals = f_vals; d1Tauxy_vals = f_vals;
tauxy_vals = f_vals;
Tauyy_vals = f_vals; d1Tauyy_vals = f_vals;
tauyy_vals = f_vals;

% Set up the doubles vectors
for d1 = 0 : N
    if (d1 == 0) || (d1 == N)
        f_vals = f_vals + (1 / 2) * a_f_consts(d1 + 1) * T_cts_mat(d1 + 1, :);
        d1f_vals = d1f_vals + (1 / 2) * a_f_consts(d1 + 1) * Tp_cts_mat(d1 + 1, :);
        d2f_vals = d2f_vals + (1 / 2) * a_f_consts(d1 + 1) * Tpp_cts_mat(d1 + 1, :);
        d3f_vals = d3f_vals + (1 / 2) * a_f_consts(d1 + 1) * Tppp_cts_mat(d1 + 1, :);
        g_vals = g_vals + (1 / 2) * a_g_consts(d1 + 1) * T_cts_mat(d1 + 1, :);
        d1g_vals = d1g_vals + (1 / 2) * a_g_consts(d1 + 1) * Tp_cts_mat(d1 + 1, :);
        d2g_vals = d2g_vals + (1 / 2) * a_g_consts(d1 + 1) * Tpp_cts_mat(d1 + 1, :);
        h_vals = h_vals + (1 / 2) * a_h_consts(d1 + 1) * T_cts_mat(d1 + 1, :);
        d1h_vals = d1h_vals + (1 / 2) * a_h_consts(d1 + 1) * Tp_cts_mat(d1 + 1, :);
        Tauxx_vals = Tauxx_vals + (1 / 2) * a_Tauxx_consts(d1 + 1) * T_cts_mat(d1 + 1, :);
        d1Tauxx_vals = d1Tauxx_vals + (1 / 2) * a_Tauxx_consts(d1 + 1) * Tp_cts_mat(d1 + 1, :);
        tauxx_vals = tauxx_vals + (1 / 2) * a_tauxx_consts(d1 + 1) * T_cts_mat(d1 + 1, :);
        Tauxy_vals = Tauxy_vals + (1 / 2) * a_Tauxy_consts(d1 + 1) * T_cts_mat(d1 + 1, :);
        d1Tauxy_vals = d1Tauxy_vals + (1 / 2) * a_Tauxy_consts(d1 + 1) * Tp_cts_mat(d1 + 1, :);
        tauxy_vals = tauxy_vals + (1 / 2) * a_tauxy_consts(d1 + 1) * T_cts_mat(d1 + 1, :);
        Tauyy_vals = Tauyy_vals + (1 / 2) * a_Tauyy_consts(d1 + 1) * T_cts_mat(d1 + 1, :);
        d1Tauyy_vals = d1Tauyy_vals + (1 / 2) * a_Tauyy_consts(d1 + 1) * Tp_cts_mat(d1 + 1, :);
        tauyy_vals = tauyy_vals + (1 / 2) * a_tauyy_consts(d1 + 1) * T_cts_mat(d1 + 1, :);
    else
        f_vals = f_vals + a_f_consts(d1 + 1) * T_cts_mat(d1 + 1, :);
        d1f_vals = d1f_vals + a_f_consts(d1 + 1) * Tp_cts_mat(d1 + 1, :);
        d2f_vals = d2f_vals + a_f_consts(d1 + 1) * Tpp_cts_mat(d1 + 1, :);
        d3f_vals = d3f_vals + a_f_consts(d1 + 1) * Tppp_cts_mat(d1 + 1, :);
        g_vals = g_vals + a_g_consts(d1 + 1) * T_cts_mat(d1 + 1, :);
        d1g_vals = d1g_vals + a_g_consts(d1 + 1) * Tp_cts_mat(d1 + 1, :);
        d2g_vals = d2g_vals + a_g_consts(d1 + 1) * Tpp_cts_mat(d1 + 1, :);
        h_vals = h_vals + a_h_consts(d1 + 1) * T_cts_mat(d1 + 1, :);
        d1h_vals = d1h_vals + a_h_consts(d1 + 1) * Tp_cts_mat(d1 + 1, :);
        Tauxx_vals = Tauxx_vals + a_Tauxx_consts(d1 + 1) * T_cts_mat(d1 + 1, :);
        d1Tauxx_vals = d1Tauxx_vals + a_Tauxx_consts(d1 + 1) * Tp_cts_mat(d1 + 1, :);
        tauxx_vals = tauxx_vals + a_tauxx_consts(d1 + 1) * T_cts_mat(d1 + 1, :);
        Tauxy_vals = Tauxy_vals + a_Tauxy_consts(d1 + 1) * T_cts_mat(d1 + 1, :);
        d1Tauxy_vals = d1Tauxy_vals + a_Tauxy_consts(d1 + 1) * Tp_cts_mat(d1 + 1, :);
        tauxy_vals = tauxy_vals + a_tauxy_consts(d1 + 1) * T_cts_mat(d1 + 1, :);
        Tauyy_vals = Tauyy_vals + a_Tauyy_consts(d1 + 1) * T_cts_mat(d1 + 1, :);
        d1Tauyy_vals = d1Tauyy_vals + a_Tauyy_consts(d1 + 1) * Tp_cts_mat(d1 + 1, :);
        tauyy_vals = tauyy_vals + a_tauyy_consts(d1 + 1) * T_cts_mat(d1 + 1, :);
    end
    
    if mod(d1+1, 10) == 0
        fprintf('Cheb. poly. comp. added: %d\n', d1+1);
    end
end

% Set up Txx, Txy and Tyy conversion
Txx_sol_full = Tauxx_sol_full;
Txy_sol_full = Tauxy_sol_full + h_sol_full;
Tyy_sol_full = Tauyy_sol_full - 2 * g_sol_full;
Txx_vals = Tauxx_vals;
Txy_vals = Tauxy_vals + h_vals;
Tyy_vals = Tauyy_vals - 2 * g_vals;

fprintf('Continuous vectors calculated.\n');

toc

% Plotting variables
LW = 'LineWidth'; IN = 'Interpreter'; LA = 'Latex'; FS = 'FontSize'; 
VA = 'VerticalAlignment';

% Figure for continuous f and g functions, along with collocation points
figure
subplot(2,2,1)
plot(x_vec_cts, f_vals, 'k')
hold on
plot(x_vec_cts, g_vals, 'b')
plot(x_vec_cts, h_vals, 'r')
plot(xvals, f_sol_full, 'xk')
plot(xvals, g_sol_full, 'ob')
plot(xvals, h_sol_full, '*r')
legend('f','g','h')
xlabel('$\eta$', IN, LA, FS, 16);
title(['No. of coll. pts = ',num2str(N + 1),', Wi = ',num2str(Wi),...
    ', $\beta$ = ',num2str(beta)], IN, LA)
hold off

fprintf('Plot 1 completed.\n');

% Figure for continuous Txx, Txy and Tyy functions, with collocation points
subplot(2,2,2)
plot(x_vec_cts, Txx_vals, 'k')
hold on
plot(x_vec_cts, Txy_vals, 'b')
plot(x_vec_cts, Tyy_vals, 'r')
plot(xvals, Txx_sol_full, 'xk')
plot(xvals, Txy_sol_full, 'ob')
plot(xvals, Tyy_sol_full, '*r')
legend('Txx','Txy','Tyy')
xlabel('$\eta$', IN, LA, FS, 16);
title(['No. of coll. pts = ',num2str(N + 1),', Wi = ',num2str(Wi),...
    ', $\beta$ = ',num2str(beta)], IN, LA)
hold off

fprintf('Plot 2 completed.\n');

% Figure for continuous Tauxx, Tauxy and Tauyy functions, with collocation points
subplot(2,2,4)
plot(x_vec_cts, Tauxx_vals, 'k')
hold on
plot(x_vec_cts, Tauxy_vals, 'b')
plot(x_vec_cts, Tauyy_vals, 'r')
plot(xvals, Tauxx_sol_full, 'xk')
plot(xvals, Tauxy_sol_full, 'ob')
plot(xvals, Tauyy_sol_full, '*r')
legend('$\mathcal{T}_{xx}$','$\mathcal{T}_{xy}$','$\mathcal{T}_{yy}$', IN, LA)
xlabel('$\eta$', IN, LA, FS, 16);
title(['No. of coll. pts = ',num2str(N + 1),', Wi = ',num2str(Wi),...
    ', $\beta$ = ',num2str(beta)], IN, LA)
hold off

fprintf('Plot 3 completed.\n');
