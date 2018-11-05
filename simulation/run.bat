@goto main

win,胜率
high,最高值
low,保底收益
level,大赚大亏程度
luck,新手中奖率
vio,是否要花时间求最优解,70min
recal,是否优先使用缓存过的参数
xman,保证每天前多少人不亏
daily,每天补贴多少钱（所有人都有可能获得）

:main
python Prize_Generator.py --win 0.6 --high 2 --low 0 --level 1 --luck 0.6 --recal
@pause
