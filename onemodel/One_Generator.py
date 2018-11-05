# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from onemodel.Para_solution import Para_solution
from onemodel.Gen_kernel import gen_kernel,gen_allowance

cachef = 'para.csv'

# 读外部参数
def get_option():
    import sys, getopt
    ss = ["win=","high=","low=","level=","luck=","vio","recal","xman=","daily="]
    opts, args = getopt.getopt(sys.argv[1:],"",ss)
    # 胜率
    _win = 0.7
    # 最高收益
    _high = 2
    # 最低收益
    _low = 0
    # 新用户中奖率
    _luck = 0.7
    # 大赚大亏程度(1~10)
    _level = 1
    # 慢慢求解
    _vio = False
    # 重算优先
    _recal = False
    # 保证前x人不亏
    _xman = 0
    # 一天固定补贴x元
    _daily = 0
    for op, value in opts:
        if op == "--win":
            _win = float(value)
        elif op == "--high":
            _high = float(value)
        elif op == "--low":
            _low = float(value)
        elif op == "--level":
            _level = float(value)
        elif op == "--luck":
            _luck = float(value)
        elif op == "--vio":
            _vio = True
        elif op == "--recal":
            _recal = True
        elif op == "--xman":
            _xman = float(value)
        elif op == "--daily":
            _daily = float(value)
    # 如果设置了保障人数，则不限制补贴上限
    if _xman > 0:
        _daily = 0
    return _win, _high, _low, _level, _luck, _vio, _recal, _xman, _daily


class One_Generator:
    '''
    输入：
        win:胜率
        high:最高收益
        low:最低收益
        luck:新用户中奖率
        level:大赚大亏程度(1~10)
        xman:保证前x人不亏
        daily:天固定补贴x元
    输出：
        符合分布的随机数
    例子：
        from One_Generator import One_Generator
        gen = One_Generator(win_p,high,low,level,luck)
        g = gen.gen_prize(number,win_time,cost,First)
    '''
    def __init__(self,m):
        win_p,high,low,level,luck,vio,recal,xman,daily,cap = m
        self.win_p = win_p
        self.high = high
        self.low = low
        self.level = level
        self.luck = luck
        self.vio = vio
        self.recal = recal # 是否必须重算
        self.xman = xman
        self.daily = daily
        self.out = 0 # 因为保证前x人不亏而导致的补贴
        self.out_c = 1
        self.kde = stats.gaussian_kde(cap) # 资金分布
        self.cal_para()


    def __del__(self):
        if hasattr(self,'kde'):
            pd.Series(self.kde.dataset[0]).to_msgpack('cap.data')
        print('Daily extra cost: %.2f'%(self.out/self.out_c))


    def cal_para(self):
        self.lucky_v = self.luck-self.win_p
        # Depending on self.high
        self.U2_h = self.high-1
        # Depending on self.level
        self.N_std = 0.05
        self.U_m = 4
        self.base = 0.1
        self.U1_p = 0.15
        self.U2_p = 0.05
        # 查缓存文件
        res = 99999
        if os.path.exists(cachef):
            cc = pd.read_csv(cachef)
            index = (cc.win_p == self.win_p) & \
                    (cc.high == self.high) & \
                    (cc.low == self.low) & \
                    (cc.level == self.level) & \
                    (cc.luck == self.luck) & \
                    (cc.xman == self.xman) & \
                    (cc.daily == self.daily)
            d = cc[['res','N_std','U_m','base','U1_p','U2_p']][index]
            if d.size != 0:
                t = d.loc[d.res.idxmin].tolist()
                (res,self.N_std,self.U_m,self.base,self.U1_p,self.U2_p) = t
                if not self.recal:
                    return
        para_solution = Para_solution(self.U2_h,self.win_p,self.low,
                                      self.lucky_v,self.level,self.xman,self.kde)
        if self.vio:
            mm = para_solution.violent_solution()
        else:
            mm = para_solution.genetic_solution()
        if mm[0]<res:
            (res,self.N_std,self.U_m,self.base,self.U1_p,self.U2_p) = mm
            self.record_para(res)


    def record_para(self,res):
        cc = pd.DataFrame(columns=[
                'win_p','high','low','level','luck','xman','daily',
                'res','N_std','U_m','base','U1_p','U2_p'
                ])
        cc.loc[0] = [self.win_p,self.high,self.low,self.level,self.luck,
              self.xman,self.daily,
              res,self.N_std,self.U_m,self.base,self.U1_p,self.U2_p]
        if os.path.exists(cachef):
            cc.to_csv(cachef,index = False,float_format = '%.4f',mode = 'a',header = False)
        else:
            cc.to_csv(cachef,index = False,float_format = '%.4f')


    def gen_prize(self,num,win_n,cost,first,cap,fre):
        '''
        输入：
        num:本轮调用次数
        win_n:当前盈利次数
        cost:当前累计额
        first:是否新用户
        cap:用户本金
        fre:用户参与次数
        '''
        m1 = (
                self.N_std,
                self.U1_p,
                self.U2_p,
                self.U_m,
                self.U2_h,
                self.base,
                )
        m2 = (
                self.win_p,
                self.low,
                self.lucky_v,
                self.xman,
                )
        m3 = (
                num,
                win_n,
                cost,
                first,
                cap,
                fre,
                )

        # 更新资金分布
        self.kde.dataset = np.column_stack((np.delete(self.kde.dataset,0,1),[cap,]))

        a = gen_allowance(self.daily,num)
        self.daily - a

        b, res = gen_kernel(m1,m2,m3)
        self.out += res
        if num == 1:
            self.out_c += 1
        return a + b
