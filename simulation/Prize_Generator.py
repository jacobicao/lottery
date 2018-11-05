# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import os
if not os.path.exists('fig'): os.mkdir('fig')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签  
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号  
from onemodel.One_Generator import One_Generator,get_option


def simu(a):
    date = pd.date_range(pd.datetime(2018, 4, 1),pd.datetime(2018, 6, 1))
    N = np.zeros(len(date)+1).astype(int)
    N[0] = 500
    ii = 1
    total = []
    m = get_option()
    cap = pd.Series(np.random.normal(10000,10,100))
    if os.path.exists('cap.data'):
        cap = pd.read_msgpack('cap.data')
    win, high, low, level, luck, vio, recal, xman, daily = m
    gen = One_Generator(m+(cap,))
    M = np.zeros(2000).astype(int)
    for d in date:
        n = np.random.poisson(10)
        N[ii] = N[ii-1] + n
        s = 0
        w = 0
        c = 0
        for q in range(N[ii]):
#            nth = np.random.uniform()*((q+1)**1)
#            if np.random.normal(0.5,nth) > 1:
#                continue
            g = gen.gen_prize(q,w,s,q>N[ii-1],cap[c%100],M[c])
            M[c] += 1
            c += 1
            s += g
            if g>1:
                w +=1
            total.append([d.date(),q,g])
        ii+=1

    data = pd.DataFrame(total,columns=['time_pay','UID','prize_v'])
    data.to_csv('simu.csv',index = False)
    date_group = data.groupby('time_pay')

    d1 = pd.datetime(2018,4,1).date()
    dd = date_group.get_group(d1).prize_v.sort_values(ascending=False).reset_index()
    del dd['index']
    dd.index=dd.index/len(dd)*100

    date_group.prize_v.agg(lambda x: x.sum()-len(x)).cumsum().plot()
    plt.title('累计补贴')
    plt.savefig('fig/累计补贴',dpi=300)
    figure, ax = plt.subplots()
    figure.suptitle('新模型收益分布')
    plt.subplot(121)
    plt.plot(dd.index,dd.prize_v)
    plt.ylim([0,3])
    plt.axhline(1.0,lw=0.7,color='k')
    plt.subplot(122)
    plt.hist(dd.prize_v,bins=50)
    plt.savefig('fig/新模型收益分布(%d)'%level,dpi=300)
#    plt.show()
    return float(dd.mean()),float(dd[dd>1].idxmin())


def main():
    import time
    start3 = time.time()
    d_mean = []
    d_win = []
    for i in range(1):
        j,k = simu(1)
        d_mean.append(j)
        d_win.append(k)
#    plt.plot(d_mean)
#    plt.figure()
#    plt.plot(d_win)
    print('Average: %.2f'%np.mean(d_mean))
    print('Winer Ratio: %.0f%%'%np.mean(d_win))
#    print('std:%.2f'%np.std(d_win))
    end=time.time()
    print('Task runs %0.2f seconds.' %(end - start3))


if __name__ == '__main__':
    main()
