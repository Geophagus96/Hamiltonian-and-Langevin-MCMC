f_class = function(s, a, b, c){
  return(a*s[1]+b*s[2]+c*s[1]*s[2])
}

grad_class = function(s, a, b, c){
  return(c(a+c*s[2], b+c*s[1]))
}

# a = 0.25*log(5)-0.25*log(3)
# b = 0.25*log(5)-0.25*log(3)
# c = 0.25*log(5)+0.25*log(3)

a = 0
b = 0
c = 0.5*log(1.5)
f = function(s){
  return(f_class(s,a,b,c))
}

#grad0 is the gradient of the energy function of the toy example
grad0 = function(s){
  return(grad_class(s,a,b,c))
}

true_prob = c(f(c(1,1)), f(c(1, -1)), f(c(-1,1)), f(c(-1, -1)))
true_prob = exp(true_prob)/sum(exp(true_prob))

default_phi = function(a,b,c){
  max_g = sum((abs(a)+abs(c))^2+(abs(b)+abs(c))^2)
  return(1/(2*sqrt(max_g)))
}


#outside HAMS is the function of inside HAMS, delta, eps, sigma corresponds to the three parameters, n is the number of samples
#Negation is incorporated, but over-relaxation is not
outsideHAMS_mod_alpha = function(delta, eps, phi, n){
  s1s = vector(length = n)
  s2s = vector(length = n)
  u1s = vector(length = n)
  u2s = vector(length = n)
  accs = vector(length = n)
  s = c(1,1)
  u = rnorm(2)
  alpha = 0.5*(grad0(c(1,1))-grad0(c(-1,-1)))
  for (i in 1:n){
    u12 = eps*u+sqrt(1-eps^2)*rnorm(2)
    y = s - delta*u12
    oldgrad = grad0(s)
    ps = exp(oldgrad-(1/(2*delta^2))*(y-1)^2)/(exp(oldgrad-(1/(2*delta^2))*(y-1)^2)+exp(-oldgrad-(1/(2*delta^2))*(y+1)^2))
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
    unew = (y-snew)/delta+phi*(newgrad-oldgrad-alpha*(snew-s))
    qup = prod(exp(oldgrad-(1/(2*delta^2))*(y-1)^2)+exp(-oldgrad-(1/(2*delta^2))*(y+1)^2))
    qdown = prod(exp(newgrad-(1/(2*delta^2))*(snew+delta*unew-1)^2)+exp(-newgrad-(1/(2*delta^2))*(snew+delta*unew+1)^2))
    log_ratio = f(snew)-f(s)+sum(newgrad*s-oldgrad*snew+phi*(unew-u12)*(oldgrad-newgrad-alpha*(s-snew)))
    q = (qup/qdown)*exp(log_ratio)
    if (is.na(qdown)){
      s1s[i] = s[1]
      s2s[i] = s[2]
      u1s[i] = -u12[1]
      u2s[i] = -u12[2]
      accs[i] = 0
    }
    else if (runif(1)<=q){
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

#Negation is incorporated, but over-relaxation is not
outsideHAMS_mod_alpha_default = function(delta, eps, n){
  s1s = vector(length = n)
  s2s = vector(length = n)
  u1s = vector(length = n)
  u2s = vector(length = n)
  accs = vector(length = n)
  s = c(1,1)
  u = rnorm(2)
  alpha = 0.5*(grad0(c(1,1))-grad0(c(-1,-1)))
  for (i in 1:n){
    u12 = eps*u+sqrt(1-eps^2)*rnorm(2)
    y = s - delta*u12
    oldgrad = grad0(s)
    ps = exp(oldgrad-(1/(2*delta^2))*(y-1)^2)/(exp(oldgrad-(1/(2*delta^2))*(y-1)^2)+exp(-oldgrad-(1/(2*delta^2))*(y+1)^2))
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
    phi = -sum((u12+(1/delta)*(s-snew))*(newgrad-oldgrad-alpha*(snew-s)))/sum((newgrad-oldgrad-alpha*(snew-s))^2)
    unew = (y-snew)/delta+phi*(newgrad-oldgrad-alpha*(snew-s))
    qup = prod(exp(oldgrad-(1/(2*delta^2))*(y-1)^2)+exp(-oldgrad-(1/(2*delta^2))*(y+1)^2))
    qdown = prod(exp(newgrad-(1/(2*delta^2))*(snew+delta*unew-1)^2)+exp(-newgrad-(1/(2*delta^2))*(snew+delta*unew+1)^2))
    log_ratio = f(snew)-f(s)+sum(newgrad*s-oldgrad*snew+phi*(unew-u12)*(oldgrad-newgrad-alpha*(s-snew)))
    q = (qup/qdown)*exp(log_ratio)
    if (is.na(qdown)){
      s1s[i] = s[1]
      s2s[i] = s[2]
      u1s[i] = -u12[1]
      u2s[i] = -u12[2]
      accs[i] = 0
    }
    else if (runif(1)<=q){
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
#outside HAMS is the function of inside HAMS, delta, eps, sigma corresponds to the three parameters, n is the number of samples
#Negation is incorporated, but over-relaxation is not
outsideHAMS_mod = function(delta, eps, phi, n){
  s1s = vector(length = n)
  s2s = vector(length = n)
  u1s = vector(length = n)
  u2s = vector(length = n)
  accs = vector(length = n)
  s = c(1,-1)
  u = rnorm(2)
  for (i in 1:n){
    u12 = eps*u+sqrt(1-eps^2)*rnorm(2)
    y = s - delta*u12
    oldgrad = grad0(s)
    ps = exp(oldgrad-(1/(2*delta^2))*(y-1)^2)/(exp(oldgrad-(1/(2*delta^2))*(y-1)^2)+exp(-oldgrad-(1/(2*delta^2))*(y+1)^2))
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
    unew = (y-snew)/delta+phi*(newgrad-oldgrad)
    qup = prod(exp(oldgrad-(1/(2*delta^2))*(y-1)^2)+exp(-oldgrad-(1/(2*delta^2))*(y+1)^2))
    qdown = prod(exp(newgrad-(1/(2*delta^2))*(snew+delta*unew-1)^2)+exp(-newgrad-(1/(2*delta^2))*(snew+delta*unew+1)^2))
    log_ratio = f(snew)-f(s)+sum(newgrad*s-oldgrad*snew+phi*(unew-u12)*(oldgrad-newgrad))
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

#inside HAMS is the function of inside HAMS, delta, eps, sigma corresponds to the three parameters, n is the number of samples
#Negation is incorporated, but over-relaxation is not
insideHAMS_mod = function(delta, eps, phi, n){
  s1s = vector(length = n)
  s2s = vector(length = n)
  u1s = vector(length = n)
  u2s = vector(length = n)
  accs = vector(length = n)
  s = c(1,1)
  u = rnorm(2)
  alpha = (grad0(c(1,1))-grad0(c(-1,-1)))/2
  
  for (i in 1:n){
    y = s+delta*(-eps*u+sqrt(1-eps^2)*rnorm(2))
    oldgrad = grad0(s)
    ps = exp(oldgrad-(1/(2*delta^2))*(y-1)^2)/(exp(oldgrad-(1/(2*delta^2))*(y-1)^2)+exp(-oldgrad-(1/(2*delta^2))*(y+1)^2))
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
    z = sqrt(1-eps^2)*rnorm(2)
    unew = (eps/delta)*(y-snew)+z+phi*(newgrad-oldgrad-alpha*(snew-s))
    zstar = unew-u-(eps/delta)*((y-snew)+(y-s))-z
    qup = prod(exp(oldgrad-(1/(2*delta^2))*(y-1)^2)+exp(-oldgrad-(1/(2*delta^2))*(y+1)^2))
    qdown = prod(exp(newgrad-(1/(2*delta^2))*(snew+delta*unew-1)^2)+exp(-newgrad-(1/(2*delta^2))*(snew+delta*unew+1)^2))
    log_ratio = f(snew)-f(s)-0.5*(sum(unew^2)-sum(u^2))
    log_ratio = log_ratio+sum(newgrad*s-oldgrad*snew)-(1/(2*delta^2))*(sum((y-s)^2)-sum((y-snew)^2))-(1/(2*(1-eps^2)))*(sum(zstar^2)-sum(z^2))
    q = (qup/qdown)*exp(log_ratio)
    if (prod(qback) == 0){
      s1s[i] = s[1]
      s2s[i] = s[2]
      u1s[i] = -u12[1]
      u2s[i] = -u12[2]
      accs[i] = 0
    }
    else if (runif(1)<=q){
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


delta = 1.2
eps = 0.9
ns = linspace(200, 1100, 10)
nrepeat = 200
# phis_results_m_alpha = array(rep(1, length(phis)*nrepeat*4), dim = c(length(phis), nrepeat, 4))
# acc_results_m_alpha = matrix(rep(1, length(phis)*nrepeat), length(phis), nrepeat)
# for(k in 1:length(phis)){
#   for (l in 1:nrepeat) {
#     outsideresults = outsideHAMS_mod_alpha(delta, eps, phis[k], n)
#     prod_est = c(mean((outsideresults$s1s==1)*(outsideresults$s2s==1)),  mean((outsideresults$s1s==1)*(outsideresults$s2s==-1)), mean((outsideresults$s1s==-1)*(outsideresults$s2s==1)), mean((outsideresults$s1s==-1)*(outsideresults$s2s==-1)))
#     acc_est = mean(outsideresults$acc==1)
#     phis_results_m_alpha[k,l,] = prod_est
#     acc_results_m_alpha[k,l] = acc_est
# 
#   }
# }
#
tv = matrix(vector(length = nrepeat*length(ns)), length(ns), nrepeat)
acc = matrix(vector(length = nrepeat*length(ns)), length(ns), nrepeat)
for (j in 1:length(ns)){
  for(l in 1:nrepeat){
    outsideresults = outsideHAMS_mod_alpha(delta, eps, 0, ns[j])
    est_prob = c(mean((outsideresults$s1s==1)*(outsideresults$s2s==1)), mean((outsideresults$s1s==1)*(outsideresults$s2s==-1)), mean((outsideresults$s1s==-1)*(outsideresults$s2s==1)), mean((outsideresults$s1s==-1)*(outsideresults$s2s==-1)))
    tv[j,l] = 0.5*sum(abs(true_prob-est_prob))
    acc[j,l] = mean(outsideresults$acc==1)
  }
}
# tvs = matrix(vector(length = nrepeat*length(phis)), length(phis), nrepeat)
# true_prob = c(0.45, 0.05, 0.05, 0.45)
# accs =  matrix(vector(length = nrepeat*length(phis)), length(phis), nrepeat)
# for (j in 1:length(phis)){
#   for(l in 1:nrepeat){
#     outsideresults = outsideHAMS_mod_alpha(delta, eps, phis[j], n)
#     est_prob = c(mean((outsideresults$s1s==1)*(outsideresults$s2s==1)), mean((outsideresults$s1s==1)*(outsideresults$s2s==-1)), mean((outsideresults$s1s==-1)*(outsideresults$s2s==1)), mean((outsideresults$s1s==-1)*(outsideresults$s2s==-1)))
#     tvs[j,l] = 0.5*sum(abs(true_prob-est_prob))
#     accs[j,l] = mean(outsideresults$acc==1)
#   }
# }
# outsideresults = outsideHAMS_mod_alpha(delta, eps, 0, n)
# 
# cat('Results corresponding to the outside-HAMS algorithm:')
# cat('Estimates of p(x1=1, x2=1):', mean((outsideresults$s1s==1)*(outsideresults$s2s==1)))
# cat('Estimates of p(x1=1, x2=-1):', mean((outsideresults$s1s==1)*(outsideresults$s2s==-1)))
# cat('Estimates of p(x1=-1, x2=1):',  mean((outsideresults$s1s==-1)*(outsideresults$s2s==1)))
# cat('Estimates of p(x1=-1, x2=-1):', mean((outsideresults$s1s==-1)*(outsideresults$s2s==-1)));
# cat('Acceptance probability:', mean(outsideresults$acc==1))

#plot(outsideresults$s1s+0.01*rnorm(n), outsideresults$s2s+0.01*rnorm(n), type='l')

# tv = vector(length = nrepeat)
# tv_default = vector(length = nrepeat)
# acc = vector(length = nrepeat)
# acc_default = vector(length = nrepeat)
# true_prob = c(0.45, 0.05, 0.05, 0.45)
# for(l in 1:nrepeat){
#   outsideresults = outsideHAMS_mod_alpha(delta, eps, 0, n)
#   est_prob = c(mean((outsideresults$s1s==1)*(outsideresults$s2s==1)), mean((outsideresults$s1s==1)*(outsideresults$s2s==-1)), mean((outsideresults$s1s==-1)*(outsideresults$s2s==1)), mean((outsideresults$s1s==-1)*(outsideresults$s2s==-1)))
#   tv[l] = 0.5*sum(abs(true_prob-est_prob))
#   acc[l] = mean(outsideresults$acc==1)
#   outsideresults = outsideHAMS_mod_alpha_default(delta, eps, n)
#   est_prob = c(mean((outsideresults$s1s==1)*(outsideresults$s2s==1)), mean((outsideresults$s1s==1)*(outsideresults$s2s==-1)), mean((outsideresults$s1s==-1)*(outsideresults$s2s==1)), mean((outsideresults$s1s==-1)*(outsideresults$s2s==-1)))
#   tv_default[l] = 0.5*sum(abs(true_prob-est_prob))
#   acc_default[l] = mean(outsideresults$acc==1)
# 
# }

