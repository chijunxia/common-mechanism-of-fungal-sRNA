import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family']='SimHei'

###########################      输出处理       ####################################################
FsRNA0 = input("Please enter the fcsv0 file name:")
# FsRNA1 = input("Please enter the fcsv1 file name:")
# FsRNA2 = input("Please enter the fcsv2 file name:")
# TsRNA = input("Please enter the tcsv file name:")
# T2sRNA = input("Please enter the tcsv file name:")
# factor = 43.39192855

#result_seq = input("Please enter the name of the result_seq  file:")
#end = input("Please enter the end_filtered file name:")
#sequence,length,Mean,STU_24,STU_48,STU_72,scale_24STU_M,scale_48STU_M,scale_72STU_M
start = time.clock()
rna0 = pd.DataFrame(pd.read_csv(FsRNA0, encoding='utf-8' ))
# rna1 = pd.DataFrame(pd.read_csv(FsRNA1, encoding='utf-8' ))
# rna2 = pd.DataFrame(pd.read_csv(FsRNA2, encoding='utf-8' ))
# rna1 = pd.DataFrame(pd.read_csv(TsRNA, encoding='utf-8' ))
# rna2 = pd.DataFrame(pd.read_csv(T2sRNA, encoding='utf-8' ))
a = list(set(rna0['len']))
# b = list(set(rna1['length']))
# c =  list(set(rna2['length']))
# b = set(rna1['sequence'])
# c = set(rna2['sequence'])
#c = set(pidata3['sequence'])
# d = b & a#|c
#
# e = b -a
x0 = list(rna0.len)
# x1 = list(rna1.length)
# x2 = list(rna2.length)
print(x0)
# print(x1)
# print(x2)
num0 = []
num1 = []
num2 = []

for i in a:
    n0 = x0.count(i)
    num0.append(n0)
print(a)
print(num0)


# for i in b:
#     n1 = x1.count(i)
#     num1.append(n1)
# print(b)
# print(num1)
#
# for i in c:
#     n2 = x2.count(i)
#     num2.append(n2)
# print(c)
# print(num2)
# print()
# print(x)
#dat = rna0[rna0.sequence.isin(list(d))]
#rna1['STU_24'] = (rna1['STUfactor_M_24'] - rna1['Mean'] )/43.39192855
#plt.xlim(15,30)
# 设置y轴的取值范围为
#plt.ylim(0, 500)
# sequence,length,Mean,24_STU*factor,48_STU*factor,72_STU*factor,72*0.3+48*0.7-24
# sequence,length,raw_count,STU_count,fSTU_count
#rna0 = rna0[rna0['length']==21]
# column_names = ['sequence', 'length','Mean', 'STUf_24','STUf_48','STUf_72','x']
# rna0.columns=column_names
#rna0 = rna0.reset_index(drop=True)      # 重建索引，为了使用赌轮法
# df2.sort_values(by=['b','a'])对df2按照b列排序后如有相同的再按照a列排序
# rna1 = rna1.sort_values(axis = 0,ascending =False ,by = ['length','sequence'])
# rna1 = rna1.reset_index(drop=True)
fig=plt.figure(figsize=(18, 5))
ax = fig.subplots(1,1)
# plt.subplot(131)
plt.xlabel("Sequence Length(Phytyphthora infestans1)")
plt.ylabel("Reads Counts(Phytyphthora infestans1)")

# plt.xlim([0.0, 1.0])
# plt.ylim([0.0,2000.0])

# data = rna1[rna1.sequence.isin(list(e))]
plt.bar(a,num0,ecolor='g',alpha=0.8,width=0.3)#累加
plt.plot(a, num0, 'r-',alpha=0.8)
#plt.plot(rna.length)
#plt.hist(rna.length,facecolor='pink',alpha=0,cumulative=True,rwidth=0.8)#累加

# dat = rna1[rna1.sequence.isin(list(d))]
# print(len(list(d)))
# print(dat)
# plt.scatter(rna0.length,rna0.STUf_24,facecolor='',edgecolors='b',marker='o', alpha=0.5)   #散点图
# plt.scatter(rna2.length,rna2.fSTU_count ,facecolor='',edgecolors='b',marker='o', alpha=0.5)   #散点图
# plt.scatter(rna1.length,rna1.STU_24*factor ,facecolor='r',marker='x', alpha=0.5)   #散点图


# plt.scatter(rna1.length,rna1.Mean ,facecolor='b',marker='o', alpha=0.5)   #散点图
#plt.scatter(rna1.length,rna1.Mean,facecolor='red',marker='x', alpha=0.5)   #散点图
#plt.scatter(rna0.length,rna0.STU_24,facecolor='red',marker='x', alpha=0.5)   #散点图
#plt.scatter(rna1.length,rna1.STU_24,facecolor='',edgecolors='b',marker='o', alpha=0.5)   #散点图
#plt.axis([-1,11,0,7])
#plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)



# plt.subplot(132)
# plt.xlabel("Sequence Length(Phytyphthora infestans2)")
# plt.ylabel("Reads Counts(Phytyphthora infestans2)")
# plt.bar(b,num1,ecolor='g',alpha=0.8,width=0.3)#累加
# plt.plot(b, num1, 'r-',alpha=0.8)

# plt.subplot(133)
# plt.xlabel("Sequence Length(solanum tuberosum,72 hpi(Phytyphthora infestans))")
# plt.ylabel("Reads Counts(solanum tuberosum,72 hpi(Phytyphthora infestans))")
# plt.bar(c,num2,ecolor='g',alpha=0.8,width=0.3)#累加
# plt.plot(c, num2, 'r-',alpha=0.8)

plt.savefig(FsRNA0+'1.png',dpi=300)#plt.savefig()将输出图形存储为文件，默认为png格式，可以通过dpi修改输出质量
plt.show()

'''
outfile = open(result_seq ,"w")
for i in result['sequence']:
    outfile.write(i + '\n')
outfile.close()

#sort.to_csv(end, index=None)
'''
end = time.clock()
print('Running dict time: %s Seconds'%(end-start))