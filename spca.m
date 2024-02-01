function [COEFF,SCORE,EXPLAINED,U,c] = spca(X,varargin)

%SPCA Sparse Principal Component Analysis using the penalised matrix
%   decomposition.
%   COEFF = SPCA(X,0.5,'K',2) returns the first two sparse principal
%   component coefficients for the N by P data matrix X. Rows of X 
%   correspond to observations and columns to variables. The two principal
%   component coefficient vectors are in the columns of COEFF. 0.5 is the
%   parameter controlling coefficient sparsity: 0 gives a fully sparse
%   model with only one non-zero coefficient per component, 1 gives no
%   sparsity. SPCA centers the data. The components are calculcated using
%   the penalised matrix decomposition model presented in Witten et al.
%   (2009)
%
%   [COEFF, SCORE] = SPCA(X,0.5,'K',2) returns the principal component
%   scores.
%
%   [COEFF, SCORE, EXPLAINED] = SPCA(X,0.5,'K',2) returns the explained
%   variances. Note that entries are NOT the proportion of all variance but
%   rather the ADDED proportion of variance explained by each component.
%   Thus the variance explained by the first component is EXPLAINED(1), the
%   variance explained by the first two components is sum(EXPLAINED(1:2))
%   etc. Calculated following Shen & Huang (2008).
%
%   [COEFF,SCORE,EXPLAINED,U] = spca(X,0.5,'K',2) returns the left side
%   factors of the covariance matrix (see Witten et al. for more details).
%
%   [COEFF,SCORE,EXPLAINED,U,c] = spca(X,0.5,'K',2) returns the sparsity
%   parameter values used.
%
%   [COEFF,~,~,~,c] = SPCA(X) returns a regularisation path for the first
%   principal component: the last dimension of COEFF corresponds to a
%   sequence of sparsity parameter values c. Thus COEFF(:,1,i) is the
%   coefficient vector for sparsity parameter c(i)
%
%   COEFF = SPCA(X,[0.1;0.2;0.3]) returns the first principal component
%   for c values [0.1;0.2;0.3].
%
%   COEFF = SPCA(X,[0.1,0.2,0.3],'K',3) returns the first three principal
%   components using c=0.1 for the first one, c=0.2 for the second etc.
%
%   COEFF = SPCA(..., 'PARAM1',val1, 'PARAM2',val2, ...) specifies optional
%   parameter name/value pairs (see below).
%
%   INPUTS:
%   X       -       N by p data matrix with observations in rows and
%                   variables in columns
%   c       -       Sparsity parameter. c=0 is full sparsity, c=1 no
%                   sparsity.
%                   If c is a single value, SPCA returns one model with
%                   the same sparsity for each component
%                   If c is a row vector, it should have the same length as
%                   the parameter 'K', and c(k) is used for the kth
%                   principal component
%                   If c is a column vector, a sequence of models (a
%                   regularisation path) is returned where the lth model
%                   uses c(l) for all components
%                   If c is a matrix, it should have 'K' columns and a
%                   sequence of models is returned where the kth component
%                   in the lth model uses c(l,k)
%   NAME/VALUE INPUTS:
%   'K'         -   How many principal components (default: 1)
%   'maxIter'   -   Maximum number of iterations for the PMDfromInit
%                   algorithm (default: 500)
%   'eps'       -   Stopping criterion for PMDfromInit (default: 1e-10)
%   OUTPUTS:
%   COEFF       -   P by K by L matrix of principal component coefficients
%                   where COEFF(p,k,l) is the coefficient for variable p in kth
%                   component using sparsity parameter c(l,k)
%   SCORE       -   N by K by L matrix of principal component scores,
%                   calculated with X centered
%   EXPLAINED   -   K by L matrix, proportion of variance explained
%   U           -   N by K by L matrix, left side factors from the matrix
%                   decomposition
%   c           -   Sparsity parameter values used
%
%   EXAMPLE:
%   load carbig;
%   data = [Displacement Horsepower Weight Acceleration MPG];
%   nans = sum(isnan(data),2) > 0;
%   [coeff,score] = spca(data(~nans,:),0.2,'K',2);

% REFERENCES
%   Witten, Daniela M., Robert Tibshirani, and Trevor Hastie. "A penalized 
%   matrix decomposition, with applications to sparse principal components 
%   and canonical correlation analysis." Biostatistics 10.3 (2009): 515-534.
%
%   Shen, Haipeng, and Jianhua Z. Huang. "Sparse principal component 
%   analysis via regularized low rank matrix approximation." Journal of 
%   multivariate analysis 99.6 (2008): 1015-1034.

% Author: T.Pusa, 2024

c = [];
K = 1;
param.maxIter = 500;
param.eps = 1e-10;

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
                case 'c'
                    c = varargin{1, i+1};
                case 'maxIter'
					param.maxIter = varargin{1, i+1};
                case 'eps'
					param.eps = varargin{1, i+1};
                otherwise
					error(['Could not recognise optional input names.' ...
                        '\nNo input named "%s"'],...
						varargin{1,i});
            end
        end
    end
end

if (K>1) && (size(c,2)>1) && (size(c,2)~=K)
    error(['Number of sparsity parameters does not match K. ' ...
        'Please provide either one c value for all components or one ' ...
        'c value per component.'])
end
if ~all(sort(c,1)==c)
    warning(['Columns of c are not in ascending order, strange things may' ...
        ' happen'])
end

[n,p] = size(X);

if isempty(c)
    c = linspace(0,1,100)';
end
c0 = c;
c = c*(sqrt(p)-1)+1;
L = size(c,1);
if size(c,2)==1
    c = repmat(c,1,K);
end

U = zeros(n,K,L);
V = zeros(p,K,L);
SCORE = zeros(n,K,L);
EXPLAINED = zeros(K,L);

X = X - mean(X,1,"omitmissing");
if any(isnan(X),"all")
    warning('Replacing missing values with 0')
    X(isnan(X)) = 0;
end

[~,~,vInit] = svds(X'*X,K);

for k=1:K
    Xdef = deflate(X,U,V,k,L);
        [U(:,k,end),V(:,k,end)] = PMDfromInit(Xdef,c(end,k),vInit(:,k),param);
    SCORE(:,k,end) = Xdef*V(:,k,end);
    EXPLAINED(k,end) = varExplained(X,V,k,L);
    for l=L-1:-1:1
        Xdef = deflate(X,U,V,k,l);
        [U(:,k,l),V(:,k,l)] = PMDfromInit(Xdef,c(l,k),V(:,k,l+1),param);
        SCORE(:,k,l) = Xdef*V(:,k,l);
        EXPLAINED(k,l) = varExplained(X,V,k,l);
    end
end

if K>1
    EXPLAINED = [EXPLAINED(1,:);diff(EXPLAINED)];
end
COEFF = V;
c = c0;

%% auxiliary functions

    function Xk = deflate(X,U,V,k,l)    
        Xk = X;   
        for kk=1:k-1
            dtmp = U(:,kk,l)'*Xk*V(:,kk,l);
            Xk = Xk - dtmp*U(:,kk,l)*V(:,kk,l)';
        end
    end

    function vexp = varExplained(X,V,k,l)
        Xk = X*V(:,1:k,l)*(V(:,1:k,l)'*V(:,1:k,l))^(-1)*V(:,1:k,l)';
        totv = trace(Xk'*Xk);
        vexp = totv/trace(X'*X);
    end

end