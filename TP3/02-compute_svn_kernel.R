source(here::here("TP3", "functions.R"))

library(magrittr)
library(data.table)

set.seed(42)
deltas <- c(3.5)
names(deltas) <- deltas
sims <- generate_data(ns = c(150, 150), 
                mus = list(c(-deltas/2, 0), 
                           c(deltas/2, 0)),
                sigma= list(diag(1, 2), 
                            diag(1, 2)))


lambdas <-  c(10, 1, 0.1, 0.001)
gammas <- c(0.01, 0.1, 1, 10, 100)
params <- CJ(lambda = lambdas, 
             gamma = gammas)

models_k <- params[, .(model = list(SVM(id ~ x + y, data = sims,  epochs = 300, 
                                        lambda = lambda,kernel = gauss_kernel(gamma)))), 
                   by = .(lambda, gamma)]

models_k[, lambda_e := factor(paste0("lambda == ", lambda))]
models_k[, gamma_e := factor(paste0("gamma == ", gamma))]

fields <- models_k[, predict_field(model[[1]], x = c(-5, 5), y = c(-3, 3), n = 40),
                   by = .(lambda_e, gamma_e)]


saveRDS(models_k, here::here("TP3", "models_r.Rds"))
saveRDS(sims,  here::here("TP3", "sims.Rds"))
saveRDS(fields,  here::here("TP3", "fields.Rds"))
