#The target two-dimensional distribution is exp(-beta*x1*x2-k*x1), the momentum grid u with probability p(u=0)=1-p
#p(u=1) = p(u=-1)=p/2
#We also represent (x1, x2) with binary number n, that is (x1=1 x2=1) is translated into n=3, (x1=0 x2=1) into n=1,
#(x1=1 x2=0) into n=2 and (x1=0 x2=0) into n=0
#The acceptance probability is calculated using the marginal distributions

#The function of translating n into x
ntox = function(n){
  if(n==0){
    return(c(-1,-1))
  }
  else if(n==1){
    return(c(-1,1))
  }
  else if(n==2){
    return(c(1,-1))
  }
  else{
    return(c(1,1))
  }
}

#The function of translating x into n
xton = function(x){
  if(x[1]==(-1)){
    if (x[2] == (-1)){
      return(0)
    }
    else{
      return(1)
    }
  }
  else{
    if (x[2]==(-1)){
      return(2)
    }
    else{
      return(3)
    }
  }
}

##The first function is with the proposal probability q1=p(x=1|y=1) and q0=p(x=0|y=0)
mc1_list = function(u, beta, k, p, n){
  u0 = -u
  #Translate n into x0 = (x1_0, x2_0)
  x0 = ntox(n)
  p1s = rep(0,2)
  #Marginal distributions of x1 and x2, p1s[1] = p(x1=1) and p1s[2] = p(x2=1)
  p1s[1] = exp(-k)/(exp(k)+exp(-k))
  p1s[2] = (exp(beta+k)+exp(-beta-k))/(exp(beta+k)+exp(-beta-k)+exp(beta-k)+exp(k-beta))
  #Computing y
  y = (x0+1)%/%2+u0
  y[y>0]=1
  y[y<=0]=0
  #The proposal probability qs
  qs = rep(0,2)
  x1 = rep(0,2)
  u1 = rep(0,2)
  #Calculate qs as p(x=1|y=1) if y=1 and p(x=0|y=0) if y=0
  for(i in (1:2)){
    if (y[i]==1){
      qs[i] = (p1s[i]*(1-p/2))/(p1s[i]*(1-p/2)+(1-p1s[i])*(p/2))
    }
    else{
      qs[i] =  ((1-p1s[i])*(1-p/2))/((1-p1s[i])*(1-p/2)+p1s[i]*(p/2))
    }
  }
  #Sampling (x1,u1) and (x2, u2)
  for (i in (1:2)){
    rands = runif(2)
    if (y[i]==1){
      if (rands[1]<=qs[i]){
        x1[i] = 1
        if (rands[2]<=((1-p)/(1-p/2))){
          u1[i] = 0
        }
        else{
          u1[i] = 1
        }
      }
      else{
        x1[i] = -1
        u1[i] = 1
      }
    }
    else{
      if (rands[1]<=qs[i]){
        x1[i] = -1 
        if (rands[2]<=((1-p)/(1-p/2))){ 
          u1[i] = 0 
        }
        else{ 
          u1[i] = -1 
        }
      }
      else{
        x1[i] = 1
        u1[i] = -1
      }
    }
  }
  #Calculating the acceptance probability
  rej_ratio = exp(-beta*x1[1]*x1[2]-k*x1[1]+beta*x0[1]*x0[2]+k*x0[1])
  pb = 1
  pf = 1
  pb = pb*exp(-k*x1[1])
  pb = pb*(exp(-beta-k*x1[2])+exp(beta+k*x1[2]))
  pf = pf*exp(-k*x0[1])
  pf = pf*(exp(-beta-k*x0[2])+exp(beta+k*x0[2]))
  rej_ratio = rej_ratio*pf/pb
  if (runif(1)<=rej_ratio){
    x = x1
    u = u1
    acc = 1
  }
  else{
    x = x0
    u = u0
    acc = 0
  }
  n = xton(x)
  return(list(n=n, u=u, acc=acc))
}

#The proposal probability is a pre-assigned value q1 and q0, the usage of all other variables is the same as
#in the previous method mc1_list
mc2_list = function(u, beta, k, p, n, q11, q10, q21, q20){
  u0 = -u
  #Translate n into x0 = (x1_0, x2_0)
  x0 = ntox(n)
  p1s = rep(0,2)
  #Marginal distributions of x1 and x2, p1s[1] = p(x1=1) and p1s[2] = p(x2=1)
  p1s[1] = exp(-k)/(exp(k)+exp(-k))
  p1s[2] = (exp(beta+k)+exp(-beta-k))/(exp(beta+k)+exp(-beta-k)+exp(beta-k)+exp(k-beta))
  #Computing y
  y = (x0+1)%/%2+u0
  y[y>0]=1
  y[y<=0]=0
  #The proposal probability qs
  x1 = rep(0,2)
  u1 = rep(0,2)
  q1s = c(q11, q21)
  q0s = c(q10, q20)
  for(i in (1:2)){
    if(y[i]==1){
      if(x0[i]==1){
        if(runif(1)<=q1s[i]){
          x1[i] = 1
          if(runif(1)<=((1-p)/(1-p/2))){
            u1[i] = 0
          }
          else{
            u1[i] = 1
          }
        }
        else{
          x1[i] = -1
          u1[i] = 1
        }
      }
      else{
        if(runif(1)<=((p1s[i]*(1-p/2))*(1-q1s[i])/((1-p1s[i])*(p/2)))){
          x1[i] = 1
          if(runif(1)<=((1-p)/(1-p/2))){
            u1[i] = 0
          }
          else{
            u1[i] = 1
          }
        }
        else{
          x1[i] = -1
          u1[i] = 1
        }
      }
    }
    else{
      if(x0[i]==(-1)){
        if(runif(1)<=q0s[i]){
          x1[i] = -1
          if(runif(1)<=((1-p)/(1-p/2))){
            u1[i] = 0
          }
          else{
            u1[i] = -1
          }
        }
        else{
          x1[i] = 1
          u1[i] = -1
        }
      }
      else{
        if(runif(1)<=(((1-p1s[i])*(1-p/2))*(1-q0s[i])/(p1s[i]*(p/2)))){
          x1[i] = -1
          if(runif(1)<=((1-p)/(1-p/2))){
            u1[i] = 0
          }
          else{
            u1[i] = -1
          }
        }
        else{
          x1[i] = 1
          u1[i] = -1
        }
      }
    }
  }
  rej_ratio = exp(-beta*x1[1]*x1[2]-k*x1[1]+beta*x0[1]*x0[2]+k*x0[1])
  pb = 1
  pf = 1
  pb = pb*exp(-k*x1[1])
  pb = pb*(exp(-beta-k*x1[2])+exp(beta+k*x1[2]))
  pf = pf*exp(-k*x0[1])
  pf = pf*(exp(-beta-k*x0[2])+exp(beta+k*x0[2]))
  rej_ratio = rej_ratio*pf/pb
  if (runif(1)<=rej_ratio){
    x = x1
    u = u1
    acc = 1
  }
  else{
    x = x0
    u = u0
    acc = 0
  }
  n = xton(x)
  return(list(n=n, u=u, acc=acc, x_1 = x1[1], x_2 = x1[2]))
}


#The third method used the uniform distribution embedding technique. The definitions of variables p1s, pys 
#are the same as mc1_list. The over-relaxation parameter with uniform distribution is denoted as alpha.

mc3_list = function(u, beta, k, p, n, alphas1, alphas0){
  u0 = -u
  x0 = ntox(n)
  p1s = rep(0,2)
  p1s[1] = exp(-k)/(exp(k)+exp(-k))
  p1s[2] = (exp(beta+k)+exp(-beta-k))/(exp(beta+k)+exp(-beta-k)+exp(beta-k)+exp(k-beta))
  y = (x0+1)%/%2+u0
  y[y>0]=1
  y[y<=0]=0
  x1 = rep(0,2)
  u1 = rep(0,2)
  pys = rep(0,2)
  alphas = rep(0,2)
  #Calculating the condition distribution pys, which is p(x=1|y)
  for (i in (1:2)){
    if (y[i]==1){
      pys[i] = (p1s[i]*(1-p/2))/(p1s[i]*(1-p/2)+(1-p1s[i])*(p/2))
      alphas[i] = alphas1[i]
    }
    else{
      pys[i] = (p1s[i]*(p/2))/(p1s[i]*(p/2)+(1-p1s[i])*(1-p/2))
      alphas[i] = alphas0[i]
    }
  }
  #The uniform distribution variable used for embedding x_0 is w0, and the uniform distribution embedded for x_1
  #is w1
  w1 = rep(0,2)
  for (i in (1:2)){
    #Generating w0
    if (x0[i]==1){
      w0 = pys[i]*runif(1)
    }
    else{
      w0 = pys[i]+(1-pys[i])*runif(1)
    }
    #Generating w1
    w1[i] = (w0+alphas[i]*runif(1))%%1
  }
  #Generating the new proposal x_1 for (x1, x2) and computing new proposal u_1 for (u1, u2).
  for (i in (1:2)){
    if (y[i]==1){
      if (w1[i]<=pys[i]){
        x1[i] = 1
        if (runif(1)<=((1-p)/(1-p/2))){
          u1[i] = 0
        }
        else{
          u1[i] = 1
        }
      }
      else{
        x1[i] = -1
        u1[i] = 1
      }
    }
    else{
      if (w1[i]>pys[i]){
        x1[i] = -1 
        if (runif(1)<=((1-p)/(1-p/2))){ 
          u1[i] = 0 
        }
        else{ 
          u1[i] = -1 
        }
      }
      else{
        x1[i] = 1
        u1[i] = -1
      }
    }
  }
  #Calculating the acceptance probability.
  rej_ratio = exp(-beta*x1[1]*x1[2]-k*x1[1]+beta*x0[1]*x0[2]+k*x0[1])
  pb = 1
  pf = 1
  pb = pb*exp(-k*x1[1])
  pb = pb*(exp(-beta-k*x1[2])+exp(beta+k*x1[2]))
  pf = pf*exp(-k*x0[1])
  pf = pf*(exp(-beta-k*x0[2])+exp(beta+k*x0[2]))
  rej_ratio = rej_ratio*pf/pb
  if (runif(1)<=rej_ratio){
    x = x1
    u = u1
    acc = 1
  }
  else{
    x = x0
    u = u0
    acc = 0
  }
  n = xton(x)
  return(list(n=n, u=u, acc=acc))
}

q1 = function(alpha, p){
  if ((alpha+p)<=1){
    if (alpha <=p){
      return((2*p-alpha)/(2*p))
    }
    else{
      return((p/(2*alpha)))
    }
  }
  else{
    if (alpha<=p){
      return((2*p-alpha)/(2*p)+(alpha+p-1)^2/(2*alpha*p))
    }
    else{
      return((p/(2*alpha))+(alpha+p-1)^2/(2*alpha*p))
    }
  }
}


u = c(0,0)
beta = 1
k = 1
p = 0.5
nsample = 10000
ns = rep(0,nsample)
x1s = rep(0, nsample)
x2s = rep(0, nsample)
us = rep(0, nsample)
acc_num = 0
n = 2
p1 = exp(-k)/(exp(k)+exp(-k))
p2 = (exp(beta+k)+exp(-beta-k))/(exp(beta+k)+exp(-beta-k)+exp(beta-k)+exp(k-beta))
q11 = p1*(1-p/2)/(p1*(1-p/2)+(1-p1)*(p/2))
q10 = ((1-p1)*(1-p/2))/((1-p1)*(1-p/2)+p1*(p/2))
q21 = p2*(1-p/2)/(p2*(1-p/2)+(1-p2)*(p/2))
q20 = ((1-p2)*(1-p/2))/((1-p2)*(1-p/2)+p2*(p/2))

q11bound = 1-(1-p1)*(p/2)/(p1*(1-p/2))
q10bound = 1-p1*(p/2)/((1-p1)*(1-p/2))
q21bound = 1-(1-p2)*(p/2)/(p2*(1-p/2))
q20bound = 1-p2*(p/2)/((1-p2)*(1-p/2))

for (i in (1:40000)){
 mclist = mc3_list(u, beta, k, p, n, c(0.5, 0.7), c(0.8, 0.4))
 u = mclist$u
 n = mclist$n
}
for (i in (1:nsample)){
  mclist = mc2_list(u, beta, k, p, n, 0.99, 0.96,0.92, 0.99)
  u = mclist$u
  n = mclist$n
  acc = mclist$acc
  acc_num = acc_num+acc
  ns[i] = n
  us[i] = u[1]
  x1s[i] = mclist$x_1
  x2s[i] = mclist$x_2
}
gap = 10
x1s_ave = rep(0, nsample/gap)
for(i in (1:(nsample/gap))){
  x1s_ave[i] = mean(x1s[((i-1)*gap+1):(i*gap)])
}
x2s_ave = rep(0, nsample/gap)
for(i in (1:(nsample/gap))){
  x2s_ave[i] = mean(x2s[((i-1)*gap+1):(i*gap)])
}
true_prob = c(exp(-beta+k), exp(beta+k), exp(beta-k),exp(-beta-k))
acc_num/nsample
table(ns)/nsample
table(us)/nsample




