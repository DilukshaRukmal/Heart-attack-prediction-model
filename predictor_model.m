%% Coursework 2023 -- Part 1

%% Main Body -- Do NOT edit

if ~exist('suppress_figures','var')
    clc; clear; close all;
    show_plots =    1;
else
    show_plots =    0;
end

load('dataset_heart_attack.mat'); % loads full data set: X, t and x_titles

% Section 1
tn =                    true_negative(0.80,0.85,0.05);

% Section 2
[X_tr,t_tr,X_te,t_te] = split_tr_te(X, t, 0.7); % tr stands for training, te for test

% Section 3
t_hat_te_sex =          X_te(:,find(x_titles=="sex"));
t_hat_te_fbs =          X_te(:,find(x_titles=="fbs"));
L_D_te_sex =            detection_error_loss(t_hat_te_sex, t_te);
L_D_te_fbs =            detection_error_loss(t_hat_te_fbs, t_te);

% Section 4
L_D_func_te_sex =       loss_func(t_hat_te_sex, t_te);
L_D_func_te_fbs =       loss_func(t_hat_te_fbs, t_te);

% Section 5
X2_tr =                 X_M(X_tr,2); % will be used in section 6
X9_te =                 X_M(X_te,9); % some other instantiation

% Section 6
theta2_ls =             LSsolver(X2_tr, t_tr);
Ngrid =                 101; % number of ponts in grid
[mFeature1,mFeature2] = meshgrid(linspace(50,250,Ngrid),linspace(100,600,Ngrid));
X_gr =                  [mFeature1(:),mFeature2(:)]; % gr for grid

t_hat2_ls_gr =          linear_combiner(X_M(X_gr,2), theta2_ls); % M=2 here

if show_plots
    figure; hold on;
    contourf(mFeature1,mFeature2,max(0,min(1,reshape(t_hat2_ls_gr,[Ngrid,Ngrid]))),'ShowText','on','DisplayName','LS solution'); inc_vec = linspace(0,1,11).'; colormap([inc_vec,1-inc_vec,0*inc_vec]);
    plot(X_te(t_te==0,1),X_te(t_te==0,2),'o','MarkerSize',6,'MarkerEdgeColor','k','MarkerFaceColor','c','DisplayName','t=0 test');
    plot(X_te(t_te==1,1),X_te(t_te==1,2),'^','MarkerSize',6,'MarkerEdgeColor','k','MarkerFaceColor','m','DisplayName','t=1 test');
    contour (mFeature1,mFeature2,max(0,min(1,reshape(t_hat2_ls_gr,[Ngrid,Ngrid]))),[0.5,0.5],'y--','LineWidth',3,'DisplayName','Decision line');
    xlabel('$x^{(1)}$','interpreter','latex'); ylabel('$x^{(2)}$','interpreter','latex'); colorbar; title('$\hat{t}_3(X|\theta_3)$','interpreter','latex'); legend show;
end

% Section 8
mse_vs_M_non_reversed = mse_vs_M( X_tr,          t_tr, X_te,          t_te);

% Section 9
X_tr_reversed = fliplr(X_tr);
X_te_reversed = fliplr(X_te);
mse_vs_M_reversed =     mse_vs_M( X_tr_reversed, t_tr, X_te_reversed, t_te);

if show_plots
    figure; hold all;
    plot(0:13, mse_vs_M_non_reversed, 'b',   'DisplayName', 'non reveresed');
    plot(0:13, mse_vs_M_reversed,     'r--', 'DisplayName', 'reveresed');
    xlabel('order $M$','interpreter','latex'); ylabel('Test loss'); title('Detection Error test loss vs. training size');
    legend show;
end

discussion();

disp('Licenses used (make sure only ''matlab'' follows):');
license('inuse')

function out = LSsolver(X,t) % Least Square solver
    out = ( X.' * X ) \ (X.' * t);
end


%% Functions -- Fill in the functions with your own code from this point

% Function 1
function tn= true_negative(sens, spec, prior)
    tn = spec*(1-prior)/((1-sens)*prior+spec*(1-prior));
end

% Function 2
function [X_tr,t_tr,X_te,t_te] = split_tr_te(X, t, eta)
    N =             length(t);
    N_tr =          round(N*eta);
    X_te =          X(1:N-N_tr,:);
	t_te =          t(1:N-N_tr,:);
    X_tr =          X(N-N_tr+1:end,:);
	t_tr =          t(N-N_tr+1:end,:);
end

% Function 3
function loss = detection_error_loss(t_hat, t)
    N = length(t);
    loss_arr = t_hat~=t;
    loss = sum(loss_arr);
    loss = loss/N;
end

% Function 4
function loss = loss_func(t_hat, t)
    N = length(t);
    loss = 0;
    for i = 1:N
        if(t_hat(i)==0 && t(i)==1)
            loss=loss+3;
        
        elseif(t_hat(i)==1 && t(i)==0)
            loss=loss+10;
            
        end
    end
    loss =  loss/N;
end

% Function 5
function out = X_M(X,M)
    X_splited = X(:,1:M);
    N = size(X,1);
    ones_col = ones(N,1);
    out = cat(2,ones_col,X_splited);
end

% Function 6
function out = linear_combiner( X ,  theta )
    out = X*theta;
end

% Function 7
function out = mse_loss(t_hat ,  t)
    out = (sum((t-t_hat).^2))/size(t,1);
end

% Function 8
function out = mse_vs_M( X_tr, t_tr, X_te, t_te)
    X = cat(1,X_tr,X_te);
    t = cat(1,t_tr,t_te);
    
    
    out =               (0:13).';
end

% Function 9
function discussion()
    disp('discussion:');
    disp('<ADD SHORT TEXT HERE.>');
end
