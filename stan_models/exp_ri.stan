//stan model: linear hierarchical regression with dependent intercept
data {
  int<lower=1> D; // number of predictors + intercept
  int<lower=0> N; // number of data
  int<lower=1> J; // number of regions
  array[N] real expenditure; // outcome
  array[N] int<lower=1, upper=J> gorx; // the region number
  array[N] row_vector[D] X; // the design matrix
}
parameters {
  vector[D] beta; // fixed intercept and slope
  vector[J] nu;   //region intercept
  real<lower=0> sigma_nu;  //region sd
  real<lower=0> sigma_res; // the residual sd
}
model {
real mu; 

//priors
  sigma_res ~ cauchy(0,100);
  sigma_nu ~ cauchy(0,100);
  nu ~ normal(0,sigma_nu); //region random effects
  beta ~ normal(0,100); 

// likelihood
  for (n in 1:N) {
    mu = X[n] * beta + nu[gorx[n]];
    expenditure[n] ~ normal(mu,sigma_res);
  }
}
generated quantities { //for use with the loo package
  vector[N] mu;
  vector[N] expenditure_rep;
  vector[N] log_lik;
  for (n in 1:N) {
    mu[n] = X[n] * beta + nu[gorx[n]];
    expenditure_rep[n] = normal_rng(mu[n],sigma_res);
    log_lik[n] = normal_lpdf(expenditure[n] | mu[n],sigma_res);
}
}
