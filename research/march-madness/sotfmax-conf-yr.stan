data {
  int<lower = 2> K; // Num of outcomes (playoff stages)
  int<lower = 0> N; // Num of observations
  int<lower = 1> D; // Num of covariates
  int<lower = 1> J; // Num of conferences 
  int<lower = 1> T; // Num of Years
  int<lower=1,upper=J> conf[N];
  int<lower=1,upper=T> year[N];
  int<lower = 1, upper = K> y[N];
  matrix[N, D] x;
}
parameters {
  matrix[D, K] beta;
  vector[J] alpha;
  vector[T] gamma;
  real mu_alpha;
  real<lower=0,upper=100> sigma_alpha;
  real mu_gamma;
  real<lower=0,upper=100> sigma_gamma;
}
model {
  matrix[N, K] x_beta = x * beta;
  
  alpha ~ normal(mu_alpha,sigma_alpha);
  gamma ~ normal(mu_gamma, sigma_gamma);
  mu_alpha ~ normal(0,1);
  sigma_alpha ~ normal(0,1);
  mu_gamma ~ normal(0,1);
  sigma_gamma ~ normal(0,1);

  to_vector(beta) ~ normal(0, 5);

  for (n in 1:N)
    y[n] ~ categorical_logit(x_beta[n]' + alpha[conf[n]] + gamma[year[n]]);
}