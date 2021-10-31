import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from bisect import bisect_left
import numpy as np
import timeit
# Please enter the sRNA file name:D:\H\CJX\CJX\Data\Sample\potato_positive_negative\Potato_negative.csv
# Please enter the end file name:D:\H\CJX\CJX\Data\Sample\potato_positive_negative\Potato_negative_abs_521.csv

"""
Basic roulette wheel selection: O(N)
"""
def basic(fitness):
    '''
    Input: a list of N fitness values (list or tuple)
    Output: selected index
    '''
    sumFits = sum(fitness)
    # generate a random number
    rndPoint = random.uniform(0, sumFits)
    # calculate the index: O(N)
    accumulator = 0.0
    for ind, val in enumerate(fitness):
        accumulator += val
        if accumulator >= rndPoint:
            return ind
def main():
    # init number of fitness values
    #N = [1,1,2,3,4,5,6,7,8,9]
    #n = int(input("Please enter the number you want to select:"))
    #for i in range(n) :
    #    print(N[basic(N)])
    #############################  根据样本分布使用轮盘赌的方式选择样本   ###############################################

    sRNA = input("Please enter the sRNA file name:")
    out = input("Please enter the end file name:")
    start = time.clock()
    RNAdata = pd.DataFrame(pd.read_csv(sRNA, encoding='utf-8'))
    # column_names = ['sequence', 'length', 'Mean','STUfactor_M_24','STUfactor_M_48','STUfactor_M_72']
    # RNAdata.columns=column_names
    # sequence,length,Mean,STU_24,STU_48,STU_72,scale_24STU_M,scale_48STU_M,scale_72STU_M
    lenRNAdata = len(RNAdata)
    #一共有多少行
    seq = []
    abfactor = 1704/lenRNAdata
    lenset = set(RNAdata["length"])             #长度统计
    #print(lenset)
    #输出{18, 19, 20, 21, 22, 23, 24, 25}
    for i in lenset:
        data = RNAdata.ix[RNAdata.length == i]  #对于每一种长度分别提取
        #print(data)
        #第一次循环 输出所有长度为18的序列和长度，一共1065个
        #第二次循环 输出所有长度为19的序列和长度
        #以此类推
        ab_num = round(len(data) * abfactor)    #提取数
       #1605*提取因子=17 提取17个长度为18的序列
        print(len(data))#1605
        print(ab_num)
        data = data.sample(frac=1)              # 打乱样本
        #只是样本打乱了 索引还是原样本的索引
        data = data.reset_index(drop=True)      # 重建索引，为了使用赌轮法
        #改成1234.....这样的索引
        index_ls = []                           #选择的索引
        for j in range(ab_num):                 #进行要提取的次数
            #a = basic(1)                #赌轮选择
            a = random.randint(0, len(data) - 1)
            while a in index_ls:                #是否重复判断，重复重新轮转
                #a = basic(1)
                a=random.randint(0, len(data) - 1)
            index_ls.append(a)                  #加入索引记录表中
        print(len(index_ls))
        dat = data[data.index.isin(index_ls)]   #提取选择到的样本
        c = list(set(dat['sequence']))          #获取样本的序列
        seq.extend(c)                           #加入到总序列中
        print("现在的序列数是："+ str(len(seq)))
    newdata = RNAdata[RNAdata.sequence.isin(seq)]
    print(len(newdata))
    # print(newdata)
    newdata.to_csv(out, index=None)



   # RNAdata = RNAdata.ix[RNAdata.fSTU_24>RNAdata.Mean]
   # print(lenset)
    #sort = RNAdata.sort_values('length', ascending=False)
    #sort = sort.reset_index(drop=True)
    #print(len(sort))
    #sort.to_csv(out, index=None)

    end = time.clock()
    print('Running dict time: %s Seconds' % (end - start))



if __name__ == "__main__":
    main()
