# Jax_implementation_learning
try to solve some of the problems given in the documentation


# Questions that need to be addressed

- What is normal distribution
- What is univariate, bivariate, and multivariate analysis
- what is bivariate normal distribution
- Marginal PDF

# Tools to be used

- Python
- JAX library
- matplotlib


# univariate vs bivariate analysis

Univariate: Using only one class of feature for generating the output

`y = theta1*x + theta2`

Where thetas are weight and bias

Bivariate: Using two different class of features for generating the output 

`z = theta1*x + theta2*y + theta3`

> This multivariate analysis help us to find the correlation between the features

# Normal distribution
Also known as gaussian distribution 
Continuous probability distribution for real-valued random variable 
Probability destity function 
![Probability destity function](https://wikimedia.org/api/rest_v1/media/math/render/svg/00cb9b2c9b866378626bcfa45c86a6de2f2b2e40)

## Examples
- Height of people
- Prices of shares in stock market
- Income distribution in ecomnomy
- Student average marks


# Question 1: Animate Bivariate normal distributon

```python
jax.random.multivariate_normal(key, mean, cov, shape=None, dtype=<class 'numpy.float64'>, method='cholesky')   -> array
```

## what is cholesky

It is a matrix decomposistion method to factorize a matrix into a product of metrices 
It is a decomposition of a Hermitian (positive-definite) matrix into the product of a lower triangular matrix and its conjugate transpose.
`A = [L][L]T` 

### Example
```
[[4, 12, -16] 
 [12, 37, -43]
 [-16, -43, 98]
]
-----------------------
[2 0 0]            [2 6 -8]
[6 1 0]     *      [0 1 5]
[-8 5 3]           [0 0 3]
```

