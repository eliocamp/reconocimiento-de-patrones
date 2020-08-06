SVM <- function(formula, data = environment(formula), batch = 1, lambda = 0.1, 
                epochs = 100, kernel = NULL) {
  data <- model.frame(formula, data)
  
  y <- matrix((as.numeric(data[, 1])-1)*2 -1, ncol = 1)
  
  # datos de predictores
  bare_data <- model.matrix(formula, data)[, -1]
  bare_data_scaled <- scale(bare_data)
  
  center <- attr(bare_data_scaled, "scaled:center")
  sd <- attr(bare_data_scaled, "scaled:scale")
  
  bare_data_scaled <- cbind(1, bare_data_scaled)
  
  if (is.null(kernel)) {
    W <- pegasos(bare_data_scaled, y, batch = batch, lambda = lambda, epochs = epochs)  
  } else {
    W <- pegasos_kernel(bare_data_scaled, y, kernel = kernel, batch = batch, 
                        lambda = lambda, epochs = epochs)  
  }
  attr(W, "levels") <- levels(data[[deparse(formula[[2]])]])
  attr(W, "formula") <- formula
  attr(W, "center") <- center
  attr(W, "sd") <- sd
  W
}

pegasos <- function(X, y, batch = 1, lambda = 0.1, epochs = 100) {
  
  W <- matrix(0, ncol = ncol(X), nrow = ncol(y))
  # W[1, 1] <- 1
  
  for (e in seq_len(epochs)) {
    nu <- 1/(lambda*e)
    W_old <- W
    for (i in seq_len(nrow(X))) {
      example <- sample(nrow(X), batch)
      pred <- W %*% t(X[example, , drop = FALSE])
      greater_one <- as.numeric((y[example, ] * pred) < 1)
      W <- (1 - 1/e)*W + colMeans(greater_one*nu*y[example, ]*X[example, ,drop = FALSE])
    }
  }
  
  structure(W, 
            class = "svm_lineal")
  
}


# https://people.csail.mit.edu/dsontag/courses/ml16/slides/lecture6_notes.pdf
pegasos_kernel <- function(X, y, kernel = gauss_kern(1), 
                           batch = 1, lambda = 0.1, epochs = 100) {
  B <- rep(0, length = nrow(X))
  for (e in seq_len(epochs)[-1]) {
    if (interactive()) message(e)
    converged <- FALSE
    
    for (i in seq_len(nrow(X))) {
      example <- sample(nrow(X), batch)
      
      Xj <- X[example, , drop = FALSE]
      
      non_zero <- B != 0
      
      if (all(!non_zero)) {
        K <- 0
      } else {
        K <- kernel(X[non_zero, , drop = FALSE], Xj)  
      }
      
      
      if (y[example, ]*1/(lambda*(e - 1))*sum(B[non_zero]*y[non_zero]*K) < 1) {
        B[example] <- B[example] + 1
        # converged <- FALSE
      }
    }
    
    if (converged) {
      break
    }
    
  }
  
  a <- 1/(lambda*epochs)*B
  
  structure(list(a = a[a != 0], 
                 y = y[a != 0],
                 X = X[a != 0, , drop = FALSE],
                 kernel = kernel,
                 SV = a != 0),
            class = "svm_kernel")
}



predict.svm_kernel <- function(object, newdata, ...) {
  formula <- attr(object, "formula", TRUE)
  formula <- delete.response(terms(formula))
  sd <- attr(object, "sd", TRUE)
  center <- attr(object, "center", TRUE)
  
  bare_data <- as.matrix(model.frame(formula, newdata, na.action = NULL))
  bare_data <- cbind(1, scale(bare_data, center = center, scale = sd))
  
  levels <- attr(object, "levels", TRUE)
  
  
  Z <- apply(bare_data, 1, function(Xi) object$y*object$a*object$kernel(object$X, matrix(c(Xi), ncol = 3)))
  
  lev <- colSums(Z)
  pred <- sign(lev)/2 + 1.5
  
  return(list(pred = factor(levels[pred], levels = levels), 
              lev = lev))
  
}


predict.svm_lineal <- function(object, newdata, ...) {
  formula <- attr(object, "formula", TRUE)
  formula <- delete.response(terms(formula))
  sd <- attr(object, "sd", TRUE)
  center <- attr(object, "center", TRUE)
  
  bare_data <- as.matrix(model.frame(formula, newdata, na.action = NULL))
  bare_data <- cbind(1, scale(bare_data, center = center, scale = sd))
  
  levels <- attr(object, "levels", TRUE)
  
  lev <- bare_data %*% t(object)
  pred <- sign(lev)/2 + 1.5
  
  return(list(pred = factor(levels[pred], levels = levels), 
              lev = lev))
}

gauss_kernel <-  function(gamma) {
  force(gamma)
 
  function(X, Y) {
    Y <- rray::rray_broadcast(Y, dim(X))
    Z <- X - Y
    
    Z <- apply(Z, 1, function(x) sum(x^2))
    exp(-gamma * Z)
  }
}

predict_field <- function(model, x, y, n = 30) {
  div <- data.table::CJ(x = seq(min(x), max(x), length.out = n), 
                        y = seq(min(y), max(y), length.out = n))
  
  div[, c("pred", "lev") := predict(model, div)]
  return(div)
}


generate_data <- function(ns, mus, sigmas) {
  
  vals <- lapply(seq_along(ns), function(i) {
    vals <- MASS::mvrnorm(ns[i], mus[[i]], sigmas[[i]])
    vals <- data.table::as.data.table(vals)
    colnames(vals) <-  c("x", "y")
    return(vals)
  })
  rbindlist(vals, idcol = "id")[, id := factor(id)]
}
