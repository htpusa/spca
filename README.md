# spca: Sparse Principal Component Analysis via penalised matrix decomposition

MATLAB implementation of the sparse principal component analysis method proposed in Witten et al. (2009).

## Quick start

```MATLAB
load spca_example
[coeff,score] = spca(X,0.5,'K',2);
```

X is a data matrix with observations in rows, the second argument is a parameter between 0 and 1 controlling coefficient vector sparsity: 0 is maximal sparsity, 1 is no sparsity. The name-value argument 'K' controls how many principal components are calculated.

## Optimising the sparsity parameter
```MATLAB
c = tunespca(X,'K',2);
[coeff,score] = spca(X,c,'K',2);
```

The function `tunespca` finds optimal values for the sparsity parameter using the missing value imputation scheme proposed in Witten et al. (2009). The parameter is optimised sequentially for each successive component, `c(1)`is the optimal value for the first component etc.

For more information about the arguments and behaviour, see
```MATLAB
help spca
```
and the example script `spca_example.m`.

## References

Witten, Daniela M., Robert Tibshirani, and Trevor Hastie. "A penalized matrix decomposition, with applications to sparse principal components and canonical correlation analysis." Biostatistics 10.3 (2009): 515-534.
