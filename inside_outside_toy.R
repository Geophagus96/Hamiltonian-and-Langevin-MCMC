#f is the energy/potential function of the toy example consisting pair of variables (s1, s2)
f = function(s){
  B = matrix(c(0, 0.5*log(9), 0.5*log(9),0),2,2)
  return(0.5*t(s)%*%B%*%s)
}

#grad0 is the gradient of the energy function of the toy example
grad0 = function(s){
  s1 = s[1]
  s2 = s[2]
  return(c(0.5*log(9)*s2, 0.5*log(9)*s1))
}

#inside HAMS is the function of inside HAMS, delta, eps, sigma corresponds to the three parameters, n is the number of samples
#Negation is incorporated, but over-relaxation is not
insideHAMS = function(delta, eps, sigma,n){
  s1s = vector(length = n)
  s2s = vector(length = n)
  u1s = vector(length = n)
  u2s = vector(length = n)
  accs = vector(length = n)
  s10 = 1
  s20 = 1
  u10 = rnorm(1)
  u20 = rnorm(1)
  s = c(s10, s20)
  u = c(u10, u20)
  
  for (i in 1:n){
    y = s+delta*(-eps*u+sqrt(1-eps^2)*rnorm(2))
    oldgrad = grad0(s)
    ps = exp(-(1/(2*sigma^2))*(1-s-sigma^2*oldgrad)^2-(1/(2*delta^2))*(y-1)^2)/(exp(-(1/(2*sigma^2))*(1-s-sigma^2*oldgrad)^2-(1/(2*delta^2))*(y-1)^2)+exp(-(1/(2*sigma^2))*(-1-s-sigma^2*oldgrad)^2-(1/(2*delta^2))*(y+1)^2))
    snew = vector(length=2)
    for (j in 1:2){
      if (runif(1) <= ps[j]){
        snew[j] = 1
      }
      else{
        snew[j] = -1
      }
    }
    newgrad = grad0(snew)
    unew = (eps/delta)*(y-snew)+sqrt(1-eps^2)*rnorm(2)
    qup = prod(exp(-(1/(2*sigma^2))*(1-s-sigma^2*oldgrad)^2-1/(2*delta^2)*(y-1)^2)+exp(-(1/(2*sigma^2))*(-1-s-sigma^2*oldgrad)^2-1/(2*delta^2)*(y+1)^2))
    qdown = prod(exp(-(1/(2*sigma^2))*(1-snew-sigma^2*newgrad)^2-1/(2*delta^2)*(y-1)^2)+exp(-(1/(2*sigma^2))*(-1-snew-sigma^2*newgrad)^2-1/(2*delta^2)*(y+1)^2))
    log_ratio = f(snew)-f(s)-1/(2*sigma^2)*(sum((s-snew-sigma^2*newgrad)^2)-sum((snew-s-sigma^2*oldgrad)^2))
    log_ratio = log_ratio-1/(2*(1-eps^2))*(sum((unew-(eps/delta)*(y-snew))^2)+sum((-u-(eps/delta)*(y-snew))^2)-sum((-u-(eps/delta)*(y-s))^2)-sum((unew-(eps/delta)*(y-s))^2))
    
    q = (qup/qdown)*exp(log_ratio)
    if (runif(1)<=q){
      s = snew
      u = unew
      s1s[i] = snew[1]
      s2s[i] = snew[2]
      u1s[i] = unew[1]
      u2s[i] = unew[2]
      accs[i] = 1
    }
    else{
      u = -u
      s1s[i] = s[1]
      s2s[i] = s[2]
      u1s[i] = -u[1]
      u2s[i] = -u[2]
      accs[i] = 0
    }
  }
  return(list(s1s=s1s, s2s=s2s, u1s = u1s, u2s = u2s, acc = accs))
  #s1s: the array of samples of s1
  #s2s: the array of samples of s2
  #u1s: the array of samples of u1, first dimension of the momentum
  #u2s: the array of samples of u2, second dimension of the momentum
  #acc: the array of acceptance or rejection
}

#inside HAMS is the function of inside HAMS, delta, eps, sigma corresponds to the three parameters, n is the number of samples
#Negation is incorporated, but over-relaxation is not
outsideHAMS = function(delta, eps, sigma, n){
  s1s = vector(length = n)
  s2s = vector(length = n)
  u1s = vector(length = n)
  u2s = vector(length = n)
  accs = vector(length = n)
  s10 = 1
  s20 = 1
  u10 = rnorm(1)
  u20 = rnorm(1)
  s = c(s10,s20)
  u = c(u10, u20)
  for (i in 1:n){
    u12 = eps*u+sqrt(1-eps^2)*rnorm(2)
    y = s - delta*u12
    oldgrad = grad0(s)
    ps = exp(-(1/(2*sigma^2))*(1-s-sigma^2*oldgrad)^2-(1/(2*delta^2))*(y-1)^2)/(exp(-(1/(2*sigma^2))*(1-s-sigma^2*oldgrad)^2-(1/(2*delta^2))*(y-1)^2)+exp(-(1/(2*sigma^2))*(-1-s-sigma^2*oldgrad)^2-(1/(2*delta^2))*(y+1)^2))
    snew = vector(length=2)
    for (j in 1:2){
      if (runif(1) <= ps[j]){
        snew[j] = 1
      }
      else{
        snew[j] = -1
      }
    }
    newgrad = grad0(snew)
    unew = (y-snew)/delta
    qup = prod(exp(-(1/(2*sigma^2))*(1-s-sigma^2*oldgrad)^2-1/(2*delta^2)*(y-1)^2)+exp(-(1/(2*sigma^2))*(-1-s-sigma^2*oldgrad)^2-1/(2*delta^2)*(y+1)^2))
    qdown = prod(exp(-(1/(2*sigma^2))*(1-snew-sigma^2*newgrad)^2-1/(2*delta^2)*(y-1)^2)+exp(-(1/(2*sigma^2))*(-1-snew-sigma^2*newgrad)^2-1/(2*delta^2)*(y+1)^2))
    log_ratio = f(snew)-f(s)-1/(2*sigma^2)*(sum((s-snew-sigma^2*newgrad)^2)-sum((snew-s-sigma^2*oldgrad)^2))
    q = (qup/qdown)*exp(log_ratio)

    if (runif(1)<=q){
      s = snew
      u = unew
      s1s[i] = snew[1]
      s2s[i] = snew[2]
      u1s[i] = unew[1]
      u2s[i] = unew[2]
      accs[i] = 1
    }
    else{
      s1s[i] = s[1]
      s2s[i] = s[2]
      u1s[i] = -u12[1]
      u2s[i] = -u12[2]
      accs[i] = 0
    }
  }
  return(list(s1s=s1s, s2s=s2s, u1s = u1s, u2s = u2s, acc = accs))
  #s1s, s2s, u1s, u2s, accs correspond to the same variable as the inside HAMS
}

delta = 0.9
eps = 0.95
sigma = 1.1
n = 100000
insideresults = insideHAMS(delta, eps, sigma, n)
outsideresults = outsideHAMS(delta, eps, sigma, n)
cat('Results corresponding to the inside-HAMS algorithm:')
cat('Estimates of p(x1=1, x2=1):', mean((insideresults$s1s==1)*(insideresults$s2s==1)))
cat('Estimates of p(x1=1, x2=-1):', mean((insideresults$s1s==1)*(insideresults$s2s==-1)))
cat('Estimates of p(x1=-1, x2=1):', mean((insideresults$s1s==-1)*(insideresults$s2s==1)))
cat('Estimates of p(x1=-1, x2=-1):', mean((insideresults$s1s==-1)*(insideresults$s2s==-1)))
cat('Acceptance probability:', mean(insideresults$acc==1))
cat('Results corresponding to the outside-HAMS algorithm:')
cat('Estimates of p(x1=1, x2=1):', mean((outsideresults$s1s==1)*(outsideresults$s2s==1)))
cat('Estimates of p(x1=1, x2=-1):', mean((outsideresults$s1s==1)*(outsideresults$s2s==-1)))
cat('Estimates of p(x1=-1, x2=1):', mean((outsideresults$s1s==-1)*(outsideresults$s2s==1)))
cat('Estimates of p(x1=-1, x2=-1):', mean((outsideresults$s1s==-1)*(outsideresults$s2s==-1)))
cat('Acceptance probability:', mean(outsideresults$acc==1))
