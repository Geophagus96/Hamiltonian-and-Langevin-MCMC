require(pracma)
#f is the energy/potential function of the toy example consisting pair of variables (s1, s2)
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
c = 0.5*log(3)

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
outsideHAMS = function(delta, eps, n){
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
    unew = (y-snew)/delta
    qup = prod(exp(oldgrad-(1/(2*delta^2))*(y-1)^2)+exp(-oldgrad-(1/(2*delta^2))*(y+1)^2))
    qdown = prod(exp(newgrad-(1/(2*delta^2))*(y-1)^2)+exp(-newgrad-(1/(2*delta^2))*(y+1)^2))
    log_ratio = f(snew)-f(s)+sum(newgrad*s-oldgrad*snew)
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

#Below are the results of outside HAMS with over-relaxation, with respect to the algorithm with over-relaxation,
#we need to first define the function of deciding forward and backward transition probabilities

transit = function(x0, x1, p){
  q = max(0, 2*p-1)
  if (x0 == 1){
    if (x1 == 1){
      return(q/p)
    }
    else{
      return(1-q/p)
    }
  }
  else{
    if (x1==1){
      return((p-q)/(1-p))
    }
    else{
      return((1+q-2*p)/(1-p))
    }
  }
}


outsideHAMS_overrelax = function(delta, eps, n){
  s1s = vector(length = n)
  s2s = vector(length = n)
  u1s = vector(length = n)
  u2s = vector(length = n)
  accs = vector(length = n)
  s = c(1,1)
  u = rnorm(2)
  for (i in 1:n){
    u12 = eps*u+sqrt(1-eps^2)*rnorm(2)
    y = s - delta*u12
    oldgrad = grad0(s)
    pfore = exp(oldgrad-(1/(2*delta^2))*(y-1)^2)/(exp(oldgrad-(1/(2*delta^2))*(y-1)^2)+exp(-oldgrad-(1/(2*delta^2))*(y+1)^2))
    snew = vector(length=2)
    for (j in 1:2){
      if (s[j] ==1){
        if (runif(1) <= (max(0,2*pfore[j]-1)/pfore[j])){
          snew[j] = 1
        }
        else{
          snew[j] = -1
        }
      }
      else{
        if (runif(1)<=(pfore[j]-max(0,2*pfore[j]-1))/(1-pfore[j])){
          snew[j] = 1
        }
        else{
          snew[j] = -1
        }
      }
    }
    newgrad = grad0(snew)
    unew = (y-snew)/delta
    pback = exp(newgrad-(1/(2*delta^2))*(y-1)^2)/(exp(newgrad-(1/(2*delta^2))*(y-1)^2)+exp(-newgrad-(1/(2*delta^2))*(y+1)^2))
    qfore = vector(length=2)
    qback = vector(length=2)

    for (j in 1:2){
      qfore[j] = transit(s[j], snew[j], pfore[j])
      qback[j] = transit(snew[j], s[j], pback[j])
    }
    log_ratio = f(snew)-f(s)-0.5*(sum(unew^2)-sum(u^2))
    q = (prod(qback)/prod(qfore))*exp(log_ratio)
    if (prod(qback) == 0){
      s1s[i] = s[1]
      s2s[i] = s[2]
      u1s[i] = -u12[1]
      u2s[i] = -u12[2]
      accs[i] = 0
    }
    else if (runif(1)<= (prod(qback)/prod(qfore))*exp(log_ratio)){
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
}


outsideHAMS_overrelax_k = function(delta, eps, k, phi, n){
  transit = function(s0, s1, k, p){
    if (p<=0.5){
      if(s0==1){
        return(ifelse((s1==1), k, 1-k))
      }
      else{
        return(ifelse((s1==1), ((1-k)*p/(1-p)), (1-2*p+k*p)/(1-p)))
      }
    }
    else{
      if(s0==(-1)){
        return(ifelse((s1==-1), k,1-k))
      }
      else{
        return(ifelse((s1==-1), ((1-k)*(1-p)/p), 1-((1-k)*(1-p)/p)))
      }
    }
  }
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
    pfore = exp(oldgrad-(1/(2*delta^2))*(y-1)^2)/(exp(oldgrad-(1/(2*delta^2))*(y-1)^2)+exp(-oldgrad-(1/(2*delta^2))*(y+1)^2))
    alpha = (grad0(c(1,1))-grad0(c(-1,-1)))/2
    
    snew = vector(length=2)
    for (j in 1:2){
      w = runif(1)
      if (pfore[j]<=0.5){
        if(s[j]==1){
          snew[j] = ifelse(w<=k, 1, -1)
        }
        else{
          snew[j] = ifelse(w<=(pfore[j]*(1-k)/(1-pfore[j])), 1, -1)
        }
      }
      else{
        if (s[j] == (-1)){
          snew[j] = ifelse(w<=k, -1, 1)
        }
        else{
          snew[j] = ifelse(w<=((1-pfore[j])*(1-k)/pfore[j]), -1, 1)
        }
      }
    }
    newgrad = grad0(snew)
    unew = (y-snew)/delta+phi*(newgrad-oldgrad-alpha*(snew-s))
    pback = exp(newgrad-(1/(2*delta^2))*(snew+delta*unew-1)^2)/(exp(newgrad-(1/(2*delta^2))*(snew+delta*unew-1)^2)+exp(-newgrad-(1/(2*delta^2))*(snew+delta*unew+1)^2))
    qfore = vector(length=2)
    qback = vector(length=2)
    
    for (j in 1:2){
      qfore[j] = transit(s[j], snew[j], k, pfore[j])
      qback[j] = transit(snew[j], s[j], k, pback[j])
    }
    log_ratio = f(snew)-f(s)-0.5*(sum(unew^2)-sum(u12^2))
    q = (prod(qback)/prod(qfore))*exp(log_ratio)
    if (runif(1)<= q){
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
}


transit_quad = function(s0, s1, a, p){
  if (p<=0.5){
    q11 = max((1-a^2)*p+(a*(a+1))/2,0)
    if(s0==1){
      return(ifelse((s1==1), q11, 1-q11))
    }
    else{
      return(ifelse((s1==1), ((1-q11)*p/(1-p)), (1-2*p+q11*p)/(1-p)))
    }
  }
  else{
    q00 = max((1-a^2)*(1-p)+(a*(a+1))/2,0)
    if(s0==(-1)){
      return(ifelse((s1==-1), q00,1-q00))
    }
    else{
      return(ifelse((s1==-1), ((1-q00)*(1-p)/p), 1-((1-q00)*(1-p)/p)))
    }
  }
}
transit_power = function(s0, s1, a, p){
  if (p<=0.5){
    q11 = p^(2/(a+1)-1)
    if(s0==1){
      return(ifelse((s1==1), q11, 1-q11))
    }
    else{
      return(ifelse((s1==1), ((1-q11)*p/(1-p)), (1-2*p+q11*p)/(1-p)))
    }
  }
  else{
    q00 = (1-p)^(2/(a+1)-1)
    if(s0==(-1)){
      return(ifelse((s1==-1), q00,1-q00))
    }
    else{
      return(ifelse((s1==-1), ((1-q00)*(1-p)/p), 1-((1-q00)*(1-p)/p)))
    }
  }
}


outsideHAMS_overrelax_a = function(delta, eps, a, phi, n, choice){

  s1s = vector(length = n)
  s2s = vector(length = n)
  u1s = vector(length = n)
  u2s = vector(length = n)
  accs = vector(length = n)
  s = c(1,-1)
  u = rnorm(2)
  if(choice==1){
    for (i in 1:n){
      u12 = eps*u+sqrt(1-eps^2)*rnorm(2)
      y = s - delta*u12
      oldgrad = grad0(s)
      pfore = exp(oldgrad-(1/(2*delta^2))*(y-1)^2)/(exp(oldgrad-(1/(2*delta^2))*(y-1)^2)+exp(-oldgrad-(1/(2*delta^2))*(y+1)^2))
      alpha = (grad0(c(1,1))-grad0(c(-1,-1)))/2
      
      snew = vector(length=2)
      for (j in 1:2){
        w = runif(1)
        if (pfore[j]<=0.5){
          q11 = max((1-a^2)*pfore[j]+(a*(a+1))/2,0)
          if(s[j]==1){
            snew[j] = ifelse(w<=q11, 1, -1)
          }
          else{
            snew[j] = ifelse(w<=(pfore[j]*(1-q11)/(1-pfore[j])), 1, -1)
          }
        }
        else{
          q00 = max((1-a^2)*(1-pfore[j])+(a*(a+1))/2,0)
          if (s[j] == (-1)){
            snew[j] = ifelse(w<=q00, -1, 1)
          }
          else{
            snew[j] = ifelse(w<=((1-pfore[j])*(1-q00)/pfore[j]), -1, 1)
          }
        }
      }
      newgrad = grad0(snew)
      unew = (y-snew)/delta+phi*(newgrad-oldgrad-alpha*(snew-s))
      pback = exp(newgrad-(1/(2*delta^2))*(snew+delta*unew-1)^2)/(exp(newgrad-(1/(2*delta^2))*(snew+delta*unew-1)^2)+exp(-newgrad-(1/(2*delta^2))*(snew+delta*unew+1)^2))
      qfore = vector(length=2)
      qback = vector(length=2)
      
      for (j in 1:2){
        qfore[j] = transit_quad(s[j], snew[j], a, pfore[j])
        qback[j] = transit_quad(snew[j], s[j], a, pback[j])
      }
      log_ratio = f(snew)-f(s)-0.5*(sum(unew^2)-sum(u12^2))
      q = (prod(qback)/prod(qfore))*exp(log_ratio)
      if (runif(1)<= q){
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
  }
  else if (choice == 2){
    for (i in 1:n){
      u12 = eps*u+sqrt(1-eps^2)*rnorm(2)
      y = s - delta*u12
      oldgrad = grad0(s)
      pfore = exp(oldgrad-(1/(2*delta^2))*(y-1)^2)/(exp(oldgrad-(1/(2*delta^2))*(y-1)^2)+exp(-oldgrad-(1/(2*delta^2))*(y+1)^2))
      alpha = (grad0(c(1,1))-grad0(c(-1,-1)))/2
      
      snew = vector(length=2)
      for (j in 1:2){
        w = runif(1)
        if (pfore[j]<=0.5){
          q11 = pfore[j]^(2/(a+1)-1)
          if(s[j]==1){
            snew[j] = ifelse(w<=q11, 1, -1)
          }
          else{
            snew[j] = ifelse(w<=(pfore[j]*(1-q11)/(1-pfore[j])), 1, -1)
          }
        }
        else{
          q00 = (1-pfore[j])^(2/(a+1)-1)
          if (s[j] == (-1)){
            snew[j] = ifelse(w<=q00, -1, 1)
          }
          else{
            snew[j] = ifelse(w<=((1-pfore[j])*(1-q00)/pfore[j]), -1, 1)
          }
        }
      }
      newgrad = grad0(snew)
      unew = (y-snew)/delta+phi*(newgrad-oldgrad-alpha*(snew-s))
      pback = exp(newgrad-(1/(2*delta^2))*(snew+delta*unew-1)^2)/(exp(newgrad-(1/(2*delta^2))*(snew+delta*unew-1)^2)+exp(-newgrad-(1/(2*delta^2))*(snew+delta*unew+1)^2))
      qfore = vector(length=2)
      qback = vector(length=2)
      
      for (j in 1:2){
        qfore[j] = transit_power(s[j], snew[j], a, pfore[j])
        qback[j] = transit_power(snew[j], s[j], a, pback[j])
      }
      log_ratio = f(snew)-f(s)-0.5*(sum(unew^2)-sum(u12^2))
      q = (prod(qback)/prod(qfore))*exp(log_ratio)
      if (runif(1)<= q){
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
  }
  return(list(s1s=s1s, s2s=s2s, u1s = u1s, u2s = u2s, acc = accs))
}

#ks = linspace(0,0.9,10)
#as = linspace(-1,1,11)
#phis = linspace(-0.5, 0.5, 11)
delta = 1.2
eps = 0.9
#n = 5000
nrepeat = 1000
# outsideresults = outsideHAMS_overrelax_a(delta, eps, 0, (-1/log(9)) ,n)
# cat('Results corresponding to the outside-HAMS algorithm:')
# cat('Estimates of p(x1=1, x2=1):', mean((outsideresults$s1s==1)*(outsideresults$s2s==1)))
# cat('Estimates of p(x1=1, x2=-1):', mean((outsideresults$s1s==1)*(outsideresults$s2s==-1)))
# cat('Estimates of p(x1=-1, x2=1):', mean((outsideresults$s1s==-1)*(outsideresults$s2s==1)))
# cat('Estimates of p(x1=-1, x2=-1):', mean((outsideresults$s1s==-1)*(outsideresults$s2s==-1)))
# cat('Acceptance probability:', mean(outsideresults$acc==1))
# plot(outsideresults$s1s[1:1000]+0.01*rnorm(1000), outsideresults$s2s[1:1000]+0.01*rnorm(1000), type='l')
# #
# tvs = matrix(vector(length = nrepeat*length(k)), length(k), nrepeat)
# true_prob = c(0.45, 0.05, 0.05, 0.45)
# accs =  matrix(vector(length = nrepeat*length(k)), length(k), nrepeat)
# for (j in 1:length(k)){
#   for(l in 1:nrepeat){
#     outsideresults = outsideHAMS_overrelax_k(delta, eps, k[j], (-1/(2*log(9))), n)
#     est_prob = c(mean((outsideresults$s1s==1)*(outsideresults$s2s==1)), mean((outsideresults$s1s==1)*(outsideresults$s2s==-1)), mean((outsideresults$s1s==-1)*(outsideresults$s2s==1)), mean((outsideresults$s1s==-1)*(outsideresults$s2s==-1)))
#     tvs[j,l] = 0.5*sum(abs(true_prob-est_prob))
#     accs[j,l] = mean(outsideresults$acc==1)
#   }
# }

# as = c(-0.2, 0.0, 0.2, 0.4)
# phis = c(-1/log(9), -1/(2*log(9)))
# tvs = array(1, dim=c(length(as),length(phis),nrepeat))
# accs = array(1, dim=c(length(as),length(phis),nrepeat))
# true_prob = c(0.45, 0.05, 0.05, 0.45)
# 
# for (i in 1:length(phis)){
#   for (j in 1:length(as)){
#     for(l in 1:nrepeat){
#       outsideresults = outsideHAMS_overrelax_a(delta, eps, as[j], phis[i], n)
#       est_prob = c(mean((outsideresults$s1s==1)*(outsideresults$s2s==1)), mean((outsideresults$s1s==1)*(outsideresults$s2s==-1)), mean((outsideresults$s1s==-1)*(outsideresults$s2s==1)), mean((outsideresults$s1s==-1)*(outsideresults$s2s==-1)))
#       tvs[j,i,l] = 0.5*sum(abs(true_prob-est_prob))
#       accs[j,i,l] = mean(outsideresults$acc==1)
#     }
#     print(tvs[j,i,1])
#   }
# }

# ns = c(1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000)
# k = c(0.3, 0.4, 0.5)
# tvs = array(1, dim=c(length(k),length(ns),nrepeat))
# accs = array(1, dim=c(length(k),length(ns),nrepeat))
# true_prob = c(0.45, 0.05, 0.05, 0.45)
# for (i in 1:length(k)){
#   for (j in 1:length(ns)){
#     for(l in 1:nrepeat){
#       outsideresults = outsideHAMS_overrelax_k(delta, eps, k[i], (-1/(2*log(9))), ns[j])
#       est_prob = c(mean((outsideresults$s1s==1)*(outsideresults$s2s==1)), mean((outsideresults$s1s==1)*(outsideresults$s2s==-1)), mean((outsideresults$s1s==-1)*(outsideresults$s2s==1)), mean((outsideresults$s1s==-1)*(outsideresults$s2s==-1)))
#       tvs[i,j,l] = 0.5*sum(abs(true_prob-est_prob))
#       accs[i,j,l] = mean(outsideresults$acc==1)
#     }
#   }
# }

# ns = linspace(200, 1100, 10)
# as = c(-0.2, -0.1, 0.0, 0.1, 0.2)
# tvs = array(1, dim=c(length(as),length(ns),nrepeat))
# accs = array(1, dim=c(length(as),length(ns),nrepeat))
# for (i in 1:length(as)){
#   for (j in 1:length(ns)){
#     for(l in 1:nrepeat){
#       outsideresults = outsideHAMS_overrelax_a(delta, eps, as[i], -default_phi(a,b,c), ns[j], choice = 2)
#       est_prob = c(mean((outsideresults$s1s==1)*(outsideresults$s2s==1)), mean((outsideresults$s1s==1)*(outsideresults$s2s==-1)), mean((outsideresults$s1s==-1)*(outsideresults$s2s==1)), mean((outsideresults$s1s==-1)*(outsideresults$s2s==-1)))
#       tvs[i,j,l] = 0.5*sum(abs(true_prob-est_prob))
#       accs[i,j,l] = mean(outsideresults$acc==1)
#     }
#   }
# }
# tvs_nomod = array(1, dim=c(length(as),length(ns),nrepeat))
# accs_nomod = array(1, dim=c(length(as),length(ns),nrepeat))
#true_prob = c(0.45, 0.05, 0.05, 0.45)
# for (i in 1:length(as)){
#   for (j in 1:length(ns)){
#     for(l in 1:nrepeat){
#       outsideresults = outsideHAMS_overrelax_a(delta, eps, as[i], 0, ns[j])
#       est_prob = c(mean((outsideresults$s1s==1)*(outsideresults$s2s==1)), mean((outsideresults$s1s==1)*(outsideresults$s2s==-1)), mean((outsideresults$s1s==-1)*(outsideresults$s2s==1)), mean((outsideresults$s1s==-1)*(outsideresults$s2s==-1)))
#       tvs_nomod[i,j,l] = 0.5*sum(abs(true_prob-est_prob))
#       accs_nomod[i,j,l] = mean(outsideresults$acc==1)
#     }
#   }
# }
# 
# as = c(-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8)
# phis = linspace(-0.5, 0.5, 11)
# tvs = array(1, dim=c(length(as),length(phis), nrepeat))
# accs = array(1, dim=c(length(as),length(phis), nrepeat))
# for (i in 1:length(as)){
#   for (j in 1:length(phis)){
#     for (l in 1:nrepeat){
#       outsideresults = outsideHAMS_overrelax_a(delta, eps, as[i], phis[j], nrepeat, choice=2)
#       est_prob = c(mean((outsideresults$s1s==1)*(outsideresults$s2s==1)), mean((outsideresults$s1s==1)*(outsideresults$s2s==-1)), mean((outsideresults$s1s==-1)*(outsideresults$s2s==1)), mean((outsideresults$s1s==-1)*(outsideresults$s2s==-1)))
#       tvs[i,j,l] = 0.5*sum(abs(true_prob-est_prob))
#       accs[i,j,l] = mean(outsideresults$acc==1)
#     }
#   }
# }

#apply(tvs, 1:2, mean)
#apply(tvs, 1:2, sd)
#tvs = rbind(tvs, tv)
#boxplot(t(tvs))
