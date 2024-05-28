// Stan model for linear regression
data {
int<lower=0> N; //number of data items
int<lower=0> D; // number of predictors
matrix[N, D] X; // predictor matrix
array[N] real expenditure; // outcome vector as an array
}
parameters {
vector[D] beta; // coefficients for predictors
real<lower=0> sigma; // SD of mean
}

model {
expenditure ~ normal(X * beta, sigma); // likelihood
beta ~ normal(0, 2.5); //priors
sigma ~ cauchy(0, 5);
}
generated quantities { //for use with the loo package
  vector[N] expenditure_rep;
  vector[N] log_lik;
  for (n in 1:N) {
    expenditure_rep[n] = normal_rng(dot_product(X[n], beta), sigma);
    log_lik[n] = normal_lpdf(expenditure[n] | dot_product(X[n], beta), sigma);
  }
}
