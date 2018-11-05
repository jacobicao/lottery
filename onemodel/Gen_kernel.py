# -*- coding: utf-8 -*-
import numpy as np
 
def gen_kernel(m1,m2,m3):
    N_std,U1_p,U2_p,U_m,U2_h,base = m1
    win_p,low,lucky_v,xman = m2
    num,win,cost,first,cap,fre = m3
    
#   J = np.random.binomial(1,0.5)*np.random.poisson(2)/10
#   B = np.random.beta(0.4,0.7)
    N = np.random.normal(0.05,N_std)
    U1 = np.random.binomial(1,U1_p)*(1-np.random.uniform()**U_m)
    U2 = np.random.binomial(1,U2_p)*np.random.uniform()*U2_h
    U3 = np.random.binomial(1,max(0,min(1,cap/5e5)))/10
    n = 1-U1+U2+U3+base+N
    n = low if n<low else n
    if first:
        n += lucky_v
    else:
        n += np.random.binomial(1,0.1)*min(0.1,0.1-0.2/(np.exp((fre-15)**2/40)))

    if num > 30:
        if cost/num > 1.01:
            n = n/1.1
        elif cost/num < 0.99:
            n = n*1.1
        if win/num > (win_p+0.01):
            n = n/1.1
        elif win/num < (win_p-0.01):
            n = n*1.1
    res = 0
    if xman!=0 and num<xman and n<1:
        res = 1-n
        n = 1
    return float(n*100//1/100),res


def gen_allowance(m,i):
    if m==0:
        return 0
    if m<0:
        return -0.01
    return np.random.uniform()*(m+5)/(m+i)//0.01/100 + (0.05 if m>0.05 else 0)
