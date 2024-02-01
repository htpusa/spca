function [optC c r2] = tunespca(X,varargin)

%TUNESPCA Find optimal sparsity parameters for spca
%   optC = TUNESPCA(X) returns the optimal sparsity parameter optC for the
%   first sparse principal component of data matrix X. Optimality is
%   determined using the missing value imputation scheme proposed in Witten
%   et al. (2009): optC is the level of sparsity that has the best
%   performance in terms of recovering censored values of X.
%
%   [optC, c] = TUNESPCA(X) returns the c values that were tested.
%
%   [optC, c, r2] = TUNESPCA(X) returns the mean squared errors for all the
%                   c values tested
%   optC = TUNESPACE(X,c) tests the c values in c (should be a vector in
%   ascending order
%   optC = TUNESPACE(X,'K',3) sequentially optimises c for the first K
%   principal components
%
%   INPUTS:
%   X       -       N by p data matrix with observations in rows and
%                   variables in columns
%   c       -       Vector of c-values to try in ascending order
%   NAME/VALUE INPUTS:
%   'K'     -       How many principal components (default: 1)
%   OUTPUTS:
%   optC    -       1 by K vector, optimal sparsity parameter values
%   c       -       Sparsity parameter values tested (same for all
%                   components)
%   r2      -       Mean squared error for all c values tested in each
%                   component
%   EXAMPLE:
%   load carbig;
%   data = [Displacement Horsepower Weight Acceleration MPG];
%   nans = sum(isnan(data),2) > 0;
%   optC = tunespca(data(~nans,:),'K',2);
%   [coeff,score] = spca(data(~nans,:),optC,'K',2);

% REFERENCES
%   Witten, Daniela M., Robert Tibshirani, and Trevor Hastie. "A penalized 
%   matrix decomposition, with applications to sparse principal components 
%   and canonical correlation analysis." Biostatistics 10.3 (2009): 515-534.

c = [];
K = 1;

if ~isempty(varargin)
    if isa(varargin{1},"double")
        c = varargin{1};
        varargin(1) = [];
    end
    if rem(size(varargin, 2), 2) ~= 0
		error('Check optional inputs.');
    else
        for i = 1:2:size(varargin, 2)
            switch varargin{1, i}
                case 'K'
					K = varargin{1, i+1};
                otherwise
					error(['Could not recognise optional input names.' ...
                        '\nNo input named "%s"'],...
						varargin{1,i});
            end
        end
    end
end

rounds = 5;
ind = crossvalind('KFold',numel(X),rounds);

if isempty(c)
    c = linspace(0,1,100)';
else
    if ~(sort(c)==c)
        warning('c is not sorted, strange things may happen')
    end
    c = c(:);
end

r2 = zeros(rounds,numel(c),K);
optC = zeros(1,K);

X = X - mean(X,1);

for k=1:K
    for r=1:rounds
        Xtmp = X;
        rm = ind==r;
        Xtmp(rm) = 0;
        [V,~,~,U] = spca(Xtmp,'c',c);
        for ci=1:numel(c)
            Xest = (U(:,1,ci)'*Xtmp*V(:,1,ci))*U(:,1,ci)*V(:,1,ci)';
            r2(r,ci,k) = sum((X(rm)-Xest(rm)).^2);
        end
    end
    r2k = mean(r2(:,:,k),1);
    [~,I] = min(r2k);
    optC(k) = c(I);
    % deflate data
    dk = U(:,1,I)'*X*V(:,1,I);
    X = X - dk*U(:,1,I)*V(:,1,I)';
end

r2 = squeeze(mean(r2,1))';