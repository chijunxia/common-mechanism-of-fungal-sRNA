import regex
import time
import pandas as pd
# 输入:  sample.csv
# 输出： sample_feature.csv
'''
# -*- coding: 提取最小自由能mfe -*-
sample_mfe = {}

import os

sample = 'MFE_dwj.txt'
file = open(sample, "r")

count = 1

key = ''
val = 0
for line in file:
    if count % 3 == 2:
        key = line[0:len(line) - 1]
    elif count % 3 == 0:
        val = line[-8:-2]
        sample_mfe[key] = float(val)
        print(val)

    count = count + 1
    #print(sample_mfe[key])
'''

# -*- coding: 提取特征 -*-

# loc = ['loc1','loc2','loc3','loc4','loc5','loc''loc''loc''loc''loc''loc''loc''loc''loc''loc''loc''loc''loc''loc''loc''loc''loc''loc''loc''loc''loc''loc''loc']

def search_motif(str, substr):
    match = regex.finditer(substr, str, overlapped=True)  # 返回序列中与子串匹配的序列的位置，并包含重叠位置
    position = [x.start() for x in match]  # 将motif出现的位置加入列表
    return (len(position))  # 输出链表的长度，即为motif出现的次数 out.writelines(q)


if __name__ == '__main__':
    mo_motif = ['A', 'T', 'G', 'C']
    colume = ['sequence', 'class', 'loc1', 'loc2', 'loc3', 'loc4', 'loc5', 'loc6', 'loc7', 'loc8', 'loc9', 'loc10',
              'loc11', 'loc12', 'loc13', 'loc14', 'loc15', 'loc16', 'loc17', 'loc18', 'loc19', 'loc20', 'loc21',
              'loc22', 'loc23', 'loc24', 'loc25', 'length', 'GC%',
              '5’mo_base', '5’di_base', '3’mo_base', '3’di_base', 'A', 'T', 'G', 'C', 'AA', 'AT', 'AG', 'AC',
              'TA', 'TT', 'TG', 'TC', 'GA', 'GT', 'GG', 'GC', 'CA', 'CT', 'CG', 'CC', 'AAA', 'AAT', 'AAG', 'AAC',
              'ATA', 'ATT', 'ATG', 'ATC', 'AGA', 'AGT', 'AGG', 'AGC', 'ACA', 'ACT', 'ACG', 'ACC', 'TAA', 'TAT', 'TAG',
              'TAC', 'TTA', 'TTT', 'TTG', 'TTC', 'TGA', 'TGT', 'TGG', 'TGC', 'TCA', 'TCT', 'TCG', 'TCC', 'GAA', 'GAT',
              'GAG', 'GAC', 'GTA', 'GTT', 'GTG', 'GTC', 'GGA', 'GGT', 'GGG', 'GGC', 'GCA', 'GCT', 'GCG', 'GCC', 'CAA',
              'CAT', 'CAG', 'CAC', 'CTA', 'CTT', 'CTG', 'CTC', 'CGA', 'CGT', 'CGG', 'CGC', 'CCA', 'CCT', 'CCG', 'CCC']
    #######################       motif生成方式       ################################

    di_motif = []
    tri_motif = []
    for i in mo_motif:
        for j in mo_motif:
            new = i + j
            di_motif.append(new)
    # print(di_motif)
    # print(len(di_motif))
    for i in di_motif:
        for j in mo_motif:
            new = i + j
            tri_motif.append(new)

    # print(tri_motif)
    # print(len(tri_motif))
    ###################################################################
   # search_motif(str,substr)
    motif_ls = mo_motif
    motif_ls.extend(di_motif)
    motif_ls.extend(tri_motif)
    # print(len(motif))
    sample = "D:\H\CJX\CJX\Data\Sample\sly_positive_negative\sample_label.csv"
    out = "D:\H\CJX\CJX\Data\Sample\sly_positive_negative\sample_label_feature_520.csv"
    start = time.clock()
    data = pd.DataFrame(pd.read_csv(sample, encoding='utf-8'))
    for i in data.index:
        print(i)
        # print(data.loc[[i]])                                      # 选出索引所在的行
        # print(data.loc[i,'sequence'])                              # 选出所在的行的某个值
        #############################################################################
        seq = data.loc[i, 'sequence']
        for a in range(1, 26):  # 输出碱基所在位置信息，没有的记为N
            if a > len(seq):
                data.loc[i, 'loc' + str(a)] = 'N'
            else:
                data.loc[i, 'loc' + str(a)] = seq[a - 1]
        ######################################################################################

        data.loc[i, 'length'] = int(len(seq))  # 长度特征
        ###############################################################################
        data.loc[i, 'GC%'] = round((100.0 * (seq.count('G') + seq.count('C')) /
                                    (seq.count('G') + seq.count('C') + seq.count('A') + seq.count('T'))), 2)  # GC含量特征
        ################################################################################
       # data.loc[i, 'MFE'] = sample_mfe[seq]  # 序列对应的自由能
        #################################################################################
        data.loc[i, '5_mo_base'] = seq[0]  # 5’第1个碱基
        data.loc[i, '5_di_base'] = seq[:2]  # 5’前2个碱基
        data.loc[i, '3_mo_base'] = seq[-1]  # 3’第1个碱基
        data.loc[i, '3_di_base'] = seq[-2:]  # 3’前2个碱基
        ##########################################################################################
        for motif in motif_ls:
            data.loc[i, motif] = search_motif((data.loc[i, 'sequence']), motif)  # motif出现频率
    ############################################################################################
    cols = list(data.columns.values)  # 将分类标记放到最后
    # print(cols)
    # print(len(cols))
    newcol = []
    newcol = cols

    # newcol.remove('class')
    # newcol.append('class')
    # print(newcol)
    print(len(newcol))

    dat = data[newcol]
    # print(data)
    dat = dat.sample(frac=1)  # 打乱样本
    dat = dat.reset_index(drop=True)  # 重建索引
    #print(dat)
    #############################################################################################c
    end = time.clock()
    dat.to_csv(out, index=None)
    print('Running dict time: %s Seconds' % (end - start))
