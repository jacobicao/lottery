# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 14:56:51 2018
"""
import random
import numpy as np
from onemodel.Gen_kernel import gen_kernel


class Para_solution:
    def __init__(self,U2_h,win_p,low,lucky_v,level,xman,kde):
        self.U2_h = U2_h
        self.win_p = win_p
        self.low = low
        self.lucky_v = lucky_v
        self.level = level
        self.xman = xman
        self.kde = kde
        self.mfun_N = 200 # 损失函数中的模拟次数
        self.domain = [(0.01, 1), (0.01, 5),(0.01,1),(0.01,0.5),(0.01,0.5)] # 解空间
    
    def m_fun(self,m,para):
        N_std,U_m,base,U1_p,U2_p = m
        U2_h,win_p,low,lucky_v,level,xman = para
        
        m1 = (N_std,U1_p,U2_p,U_m,U2_h,base)
        m2 = (win_p,low,lucky_v,xman)
        
        #    a = np.array([gen_kernel(m1,m2,m3) for x in range(500)])
        w,s = 0,0
        t = []
        tt = self.kde.resample(self.mfun_N)[0]
        bb = np.random.binomial(1,0.02,self.mfun_N)
        ee = np.random.beta(0.5,0.7,self.mfun_N)*100
        for q in range(self.mfun_N):
            cap = tt[q]
            first = bb[q]
            fre = ee[q]
            m3 = (q,w,s,first,cap,fre)
            g,r = gen_kernel(m1,m2,m3)
            s += g
            if g>1:
                w +=1
            t.append(g)
        a = np.array(t)
        loss = float(
                (a.mean()*100-100)**4
                +1000*((a>1).sum()/len(a) - win_p)**2
                +0.6**level*(50*(a[a>1.2].sum()-1.2*len(a[a>1.2]))+U_m*50)
                )
        return [loss,m]

    def loss_fun(self,m):
        para = (self.U2_h,self.win_p,self.low,self.lucky_v,self.level,self.xman)
        return self.m_fun(m,para)

    def violent_solution(self):
        # 暴力求解
        n = 20
        x = [((a+1)/(10*n),(b+1)/n*5,(c+1)/n,(d+1)/(4*n),(e+1)/(4*n))
            for a in range(n)
            for b in range(n)
            for c in range(n)
            for d in range(n)
            for e in range(n)]
        from multiprocessing import Pool,cpu_count
        p = Pool(cpu_count())
        res = p.map(self.loss_fun, x)
        if len(res)==0:
            print('ERROR: No solution found!')
            return (0.05,4,0.07,0.15,0.05)
        res.sort()
        print('res:%.8f\nsol:%s'%(res[0][0],np.array(res[0][1]).round(4)))
        return [res[0][0]]+res[0][1]

    def genetic_solution(self):
        # 遗传算法
        from multiprocessing import Pool,cpu_count
        cup_ct = cpu_count()
        p = Pool(cup_ct)
        result = p.map(self.geneticoptimize, [self.domain,]*cup_ct)
        result.sort()
#        result = [self.geneticoptimize(self.domain)]
        print('res:%.8f\nsol:%s'%(result[0][0],np.array(result[0][1]).round(4)))
        return [result[0][0]]+result[0][1]

    def gradient_solution(self):
        # 梯度下降算法 不使用
        from scipy import optimize
        x = (0.05,4,0.07,0.15,0.05)
        bnds = ((0.001, 1), (0, 5),(0,1),(0,0.5),(0,0.5))
        a = optimize.minimize(self.loss_fun,x,method='L-BFGS-B', bounds=bnds)
        print(a)
        return [self.loss_fun(a.x)]+list(a.x)

    def geneticoptimize(self, domain, popsize=50, step=0.01, mutprob=0.2, elite=0.2, maxiter=100):
        # 变异操作的函数
        def mutate(vec):
            i = random.randint(0, len(domain) - 1)
            res = []
            if random.random() < 0.5 and vec[i] > domain[i][0]:
                res = vec[0:i] + [vec[i] - step] + vec[i + 1:]
            elif vec[i] < domain[i][1]:
                res = vec[0:i] + [vec[i] + step] + vec[i + 1:]
            else:
                res = vec
            return res

        # 交叉操作的函数（单点交叉）
        def crossover(r1, r2):
            i = random.randint(0, len(domain) - 1)
            return r1[0:i] + r2[i:]

        pop = []
        for i in range(popsize):
            vec = [random.uniform(domain[i][0], domain[i][1]) for i in range(len(domain))]
            pop.append(vec)
            topelite = int(elite * popsize)
        for i in range(maxiter):
            if [] in pop:
                print("***")
            try:
                scores = [self.loss_fun(v) for v in pop]
            except Exception as args:
                print(i,"pop!!",args)
            scores.sort()
            ranked = [v for (s, v) in scores]

            # 优质解遗传到下一代
            pop = ranked[0: topelite]
            # 如果当前种群数量小于既定数量，则添加变异和交叉遗传
            while len(pop) < popsize:
                # 随机数小于 mutprob 则变异，否则交叉
                if random.random() < mutprob:
                    c = random.randint(0, topelite)
                    if len(ranked[c]) != len(domain):
                        continue
                    temp = mutate(ranked[c])
                    if temp == []:
                        print("******", ranked[c])
                    else:
                        pop.append(temp)

                else:
                    c1 = random.randint(0, topelite)
                    c2 = random.randint(0, topelite)
                    pop.append(crossover(ranked[c1], ranked[c2]))
            # 输出当前种群中代价最小的解
#            print(scores[0][1], "代价：", scores[0][0])
#        print('res:%s'%list(map(lambda x:round(x,2),scores[0][1])))
#        print("遗传算法求得的最小代价：", scores[0][0])
        return scores[0]
