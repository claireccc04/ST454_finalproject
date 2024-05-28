//stan model: linear hierarchical regression
data {
  int<lower=1> D; // number of predictors + intercept
  int<lower=0> N; // number of data
  int<lower=1> J; // number of regions
  array[N] real expenditure; // outcome
  array[N] int<lower=1, upper=J> gorx; // the region number
  array[N] row_vector[D] X; // the design matrix
  array[N] real income; // income predictor
}
parameters {
  row_vector[D] beta; // the fixed effects 
  real<lower=0> sigma; // the sd for fixed effect
  real<lower=0> sigma_income; // the sd for income random effect
  array[J] real beta_income; // the random effects for income
  real<lower=0> sigma_res; // the residual sd
}
model {
//priors
  beta ~ normal(0,10);
  sigma ~ cauchy(0,5);
  sigma_income ~ cauchy(0,5);
  beta_income ~ normal(0,100);
  sigma_res ~ cauchy(0,5);
// model for the random effects
  for(j in 1:J){
    income ~ normal(beta_income[j], sigma_income);
    }
// likelihood
  for (n in 1:N) {
    real mu;
    mu = dot_product(X[n], beta) + income[n] * beta_income[gorx[n]];
    expenditure[n] ~ normal(mu,sigma_res);
  }
}
generated quantities { //for use with the loo package
  vector[N] expenditure_rep;
  vector[N] log_lik;
  for (n in 1:N) {
    real mu;
    mu = dot_product(X[n], beta) + income[n] * beta_income[gorx[n]];
    expenditure_rep[n] = normal_rng(mu, sigma_res);
    log_lik[n] = normal_lpdf(expenditure[n] | mu,sigma_res);
}
}
