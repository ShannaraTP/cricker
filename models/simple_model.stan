data {
  int<lower=1> N_mrna;
  int<lower=1> N_cond;
  matrix[N_mrna, N_cond] PSI_r;
  matrix[N_mrna, N_cond] PSI_p;
  real<lower=1> scaling_factor;

  // if we should evaluate the likelihood
  int<lower=0, upper=1> likelihood;
}

transformed data {  
  // The observed data rows are nearly but not quite 1, we have to adjust it
  matrix[N_mrna, N_cond] PSI_r_closed;
  matrix[N_mrna, N_cond] PSI_p_closed;
  for (cond in 1:N_cond) {
    PSI_r_closed[:, cond] = PSI_r[:, cond] / sum(PSI_r[:, cond]);
    PSI_p_closed[:, cond] = PSI_p[:, cond] / sum(PSI_p[:, cond]);
  }
}

parameters {
  vector[N_mrna] z_delta;
  array[N_cond] simplex[N_mrna] r;
  array[N_cond] simplex[N_mrna] p;
  real<lower=0> alpha_r;
  real<lower=0> alpha_p;
}

transformed parameters {
  vector[N_mrna] delta = 2 ^ z_delta;
  matrix[N_cond, N_mrna] psi_hat_r; 
  matrix[N_cond, N_mrna] psi_hat_p; 
  for (cond in 1:N_cond) {
    psi_hat_r[cond] = to_row_vector(r[cond] * alpha_r * scaling_factor);
    psi_hat_p[cond] = to_row_vector(p[cond] * alpha_p * scaling_factor);
  }
}

model {
  z_delta ~ normal(0, 1.2);
  for (cond in 1:N_cond) {
     r[cond] ~ dirichlet(rep_row_vector(1, N_mrna));
     p[cond] ~ dirichlet((r[cond] * scaling_factor) .* delta);
  }
  alpha_r ~ inv_gamma(0.5, 1);
  alpha_p ~ inv_gamma(0.5, 1);

  // likelihood
  if (likelihood == 1) {
    for (cond in 1:N_cond) {
       PSI_r_closed[:, cond] ~ dirichlet(psi_hat_r[cond]);
       PSI_p_closed[:, cond] ~ dirichlet(psi_hat_p[cond]);
    } 
  }
}
