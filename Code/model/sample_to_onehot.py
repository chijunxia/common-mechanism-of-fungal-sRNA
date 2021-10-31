# Please enter the sample csv file name:D:\H\CJX\CJX\Data\Sample\sly_positive_negative\sample_label_feature_520.csv
# Please enter the out csv file name:D:\H\CJX\CJX\Data\Sample\sly_positive_negative\sample_label_feature_520_onehot.csv
import regex
import time
import pandas as pd
import numpy as np
# sequence,loc1,loc2,loc3,loc4,loc5,loc6,loc7,loc8,loc9,loc10,loc11,loc12,loc13,loc14,loc15,loc16,loc17,loc18,loc19,loc20,
# loc21,loc22,loc23,loc24,loc25,length,GC%,MFE,5_mo_base,5_di_base,3_mo_base,3_di_base,
# A,T,G,C,AA,AT,AG,AC,TA,TT,TG,TC,GA,GT,GG,GC,CA,CT,CG,CC,AAA,AAT,AAG,AAC,ATA,ATT,ATG,ATC,
# AGA,AGT,AGG,AGC,ACA,ACT,ACG,ACC,TAA,TAT,TAG,TAC,TTA,TTT,TTG,TTC,TGA,TGT,TGG,TGC,TCA,TCT,
# TCG,TCC,GAA,GAT,GAG,GAC,GTA,GTT,GTG,GTC,GGA,GGT,GGG,GGC,GCA,GCT,GCG,GCC,CAA,CAT,CAG,CAC,
# CTA,CTT,CTG,CTC,CGA,CGT,CGG,CGC,CCA,CCT,CCG,CCC,class
# sample = input("Please enter the sample csv file name:")

# di_base = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
mo_hot = { 'N':[0,0,0,0],
           'A':[0,0,0,1],
           'C':[0,0,1,0],
           'G':[0,1,0,0],
           'T':[1,0,0,0]
           }
di_hot = {
    'AA': [0, 0, 0, 1, 0, 0, 0, 1], 'AC': [0, 0, 0, 1, 0, 0, 1, 0],'AG': [0, 0, 0, 1, 0, 1, 0, 0], 'AT': [0, 0, 0, 1, 1, 0, 0, 0],
    'CA': [0, 0, 1, 0, 0, 0, 0, 1], 'CC': [0, 0, 1, 0, 0, 0, 1, 0],'CG': [0, 0, 1, 0, 0, 1, 0, 0], 'CT': [0, 0, 1, 0, 1, 0, 0, 0],
    'GA': [0, 1, 0, 0, 0, 0, 0, 1], 'GC': [0, 1, 0, 0, 0, 0, 1, 0],'GG': [0, 1, 0, 0, 0, 1, 0, 0], 'GT': [0, 1, 0, 0, 1, 0, 0, 0],
    'TA': [1, 0, 0, 0, 0, 0, 0, 1], 'TC': [1, 0, 0, 0, 0, 0, 1, 0],'TG': [1, 0, 0, 0, 0, 1, 0, 0], 'TT': [1, 0, 0, 0, 1, 0, 0, 0]
          }
# st_hot = { 'N':[0,0,0],
#            '(':[0,0,1],
#            ')':[0,1,0],
#            '.':[1,0,0]
#            }
###################################################生成字典
# base = ['A','C','G','T']
# di_base =[]
# for i in base:
#     for j in base:
#         ls =[]
#         ls.extend(mo_hot[i])
#         ls.extend(mo_hot[j])
#         di_hot[i+j] = ls #.extend(mo_hot[j])
# print(di_hot)
########################################################

# sample = "sample_5_old_feature_test.csv"
sample = input("Please enter the sample csv file name:")
out = input("Please enter the out csv file name:")
# out = input("Please enter the out csv file name:")
start = time.clock()
data = pd.DataFrame(pd.read_csv(sample, encoding='utf-8'))
dat = pd.DataFrame()
# data = data.sample(frac=1)              # 打乱样本
# data['name'].str.split('|',expand=True)
print(data.dtypes)
names = list(data.columns)

for name in names:
    newnames = list(dat.columns)
    if 'loc' in name:
        if 'st' in name:
            ls = []
            for i in data.index:
                #data.ix[i, name] = str(st_hot[str(data.ix[i, name])]).strip('[]')  # 将对应的原来的二级结构换位向量，逗号隔开的
                ls = [name + '_1', name + '_2', name + '_3']
            newnames.extend(ls)  # 名字的增加的部分
            dat = pd.concat([dat, data[name].str.split(r',', expand=True)], axis=1)  # 原来的加上修改的dataframe
            dat.columns = newnames
            continue
        else:
            ls = []
            for i in data.index:
                data.ix[i, name] = str(mo_hot[str(data.ix[i, name])]).strip('[]')  # 将对应的原来的碱基换位向量，逗号隔开的
                ls = [name + '_1', name + '_2', name + '_3', name + '_4']
            newnames.extend(ls)  # 名字的增加的部分
            dat = pd.concat([dat, data[name].str.split(r',', expand=True)], axis=1)  # 原来的加上修改的dataframe
            dat.columns = newnames
        continue
    if 'mo_base' in name:
        ls = []
        for i in data.index:
            data.ix[i, name] = str(mo_hot[str(data.ix[i, name])]).strip('[]')
            ls = [name+'_1',name+'_2',name+'_3',name+'_4']
        newnames.extend(ls)
        dat = pd.concat([dat,data[name].str.split(r',', expand=True)], axis=1)
        dat.columns = newnames
        continue
    if 'di_base' in name:
        ls = []
        for i in data.index:
            data.ix[i, name] = str(di_hot[str(data.ix[i, name])]).strip('[]')
            ls = [name+'_1',name+'_2',name+'_3',name+'_4',name+'_5',name+'_6',name+'_7',name+'_8']
        newnames.extend(ls)
        dat = pd.concat([dat,data[name].str.split(r',', expand=True)], axis=1)
        dat.columns = newnames
        continue
    # if 'st_loc' in name:
    #     ls = []
    #     for i in data.index:
    #         data.ix[i, name] = str(st_hot[str(data.ix[i, name])]).strip('[]')  # 将对应的原来的二级结构换位向量，逗号隔开的
    #         ls = [name + '_1', name + '_2', name + '_3']
    #     newnames.extend(ls)  # 名字的增加的部分
    #     dat = pd.concat([dat, data[name].str.split(r',', expand=True)], axis=1)  # 原来的加上修改的dataframe
    #     dat.columns = newnames
    #     continue
    dat = pd.concat([dat, data[name]], axis=1)


print(dat)
# print(dat.dtypes)
da = dat.drop(['sequence'], axis=1, inplace=False)
#da = da.drop(['second_structure'], axis=1, inplace=False)
da = pd.DataFrame(da,dtype=np.float)
# print(da.dtypes)
ndata = pd.concat([data['sequence'], da], axis=1)
# ndata = pd.concat([data['sequence'], da], axis=1)
print(ndata.dtypes)
ndata.to_csv(out,index=None)










end = time.clock()
# data.to_csv(out, index=None)
print('Running dict time: %s Seconds' % (end - start))