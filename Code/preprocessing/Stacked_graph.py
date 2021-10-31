import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# N = 214

# S = (52, 49, 48, 47, 44, 43, 41, 41, 40, 38, 36, 31, 29)
# C = (38, 40, 45, 42, 48, 51, 53, 54, 57, 59, 57, 64, 62)

rna1 = pd.DataFrame(pd.read_csv('D:\G\chi_paper\Exit_after_before\Mo\Os_sort_by_seq.csv', encoding='utf-8' ))
rna2 = pd.DataFrame(pd.read_csv('D:\G\chi_paper\Exit_after_before\Mo\dwj_inner_214_sort_by_seq.csv', encoding='utf-8' ))
rna3 = pd.DataFrame(pd.read_csv('D:\G\chi_paper\Exit_after_before\Pi\po_sort_seq.csv', encoding='utf-8' ))
rna4 = pd.DataFrame(pd.read_csv('D:\G\chi_paper\Exit_after_before\Pi\yimei_inner_248_sort_seq.csv', encoding='utf-8' ))
rna5 = pd.DataFrame(pd.read_csv('D:\G\chi_paper\Exit_after_before\Bc\sly_after_175__sort_seq.csv', encoding='utf-8' ))
rna6 = pd.DataFrame(pd.read_csv('D:\G\chi_paper\Exit_after_before\Bc\sly_before_after_175_sort_seq.csv', encoding='utf-8' ))
# a = set(rna1['seq'])
# b = set(rna2['seq'])
# c = set(rna3['seq'])
# d = set(rna4['seq'])
# e = set(rna5['seq'])
# f = set(rna6['seq'])
fig=plt.figure(figsize=(15, 5))
ax = fig.subplots(1,3)
plt.subplot(131)
N = len(rna1)
d = []
for i in range(0, len(rna1)):
    sum = rna1.nor+ rna2.count_x
    d.append(sum)
# M = (10, 11, 7, 11, 8, 6, 6, 5, 3, 3, 7, 5, 9)
# menStd = (2, 3, 4, 1, 2)
# womenStd = (3, 5, 2, 3, 3)
ind = np.arange(N)  # the x locations for the groups
width = 1 # the width of the bars: can also be len(x) sequence

# p1 = plt.bar(ind, rna1.nor, width, color='#d62728')  # , yerr=menStd)
# p2 = plt.bar(ind, rna2.count_x, width, bottom=rna1.nor)  # , yerr=womenStd)
p1 = plt.bar(ind, rna2.count_x,width , color='#d62728')  # , yerr=menStd)
p2 = plt.bar(ind, rna1.nor, width, bottom=rna2.count_x)  # , yerr=womenStd)
# p3 = plt.bar(ind, M, width, bottom=d)

plt.ylabel('Count')
plt.title('Magnaporthe oryzae')
# plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13'))
# plt.yticks(np.arange(0, 6000, 500))
plt.legend((p1[0], p2[0]), ('Before', 'After'))
# # plt.legend((p1[0], p2[0], p3[0]), ('S', 'C', 'M'))
# plt.savefig('after and before_90000_Os.jpg',dpi=300)
# plt.show()

plt.subplot(132)
N = len(rna6)
d = []
for i in range(0, len(rna6)):
    sum = rna6.before + rna6.after
    d.append(sum)
# M = (10, 11, 7, 11, 8, 6, 6, 5, 3, 3, 7, 5, 9)
# menStd = (2, 3, 4, 1, 2)
# womenStd = (3, 5, 2, 3, 3)
ind = np.arange(N)  # the x locations for the groups
width = 1 # the width of the bars: can also be len(x) sequence

# p1 = plt.bar(ind, rna1.nor, width, color='#d62728')  # , yerr=menStd)
# p2 = plt.bar(ind, rna2.count_x, width, bottom=rna1.nor)  # , yerr=womenStd)
p1 = plt.bar(ind, rna6.before,width , color='#d62728')  # , yerr=menStd)
p2 = plt.bar(ind, rna6.after, width, bottom=rna6.before)  # , yerr=womenStd)
# p3 = plt.bar(ind, M, width, bottom=d)

plt.ylabel('Count')
plt.title('Botrytis cinerea')
# plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13'))
# plt.yticks(np.arange(0, 150, 15))
plt.legend((p1[0], p2[0]), ('Before', 'After'))
# # plt.legend((p1[0], p2[0], p3[0]), ('S', 'C', 'M'))
# plt.savefig('after and before_150_sly.jpg',dpi=300)
# plt.show()


plt.subplot(133)
N = len(rna4)
d = []
for i in range(0, len(rna4)):
    sum = rna4.nor_x[i] + rna4.nor_y[i]
    d.append(sum)
# M = (10, 11, 7, 11, 8, 6, 6, 5, 3, 3, 7, 5, 9)
# menStd = (2, 3, 4, 1, 2)
# womenStd = (3, 5, 2, 3, 3)
ind = np.arange(N)  # the x locations for the groups
width = 1  # the width of the bars: can also be len(x) sequence

# p1 = plt.bar(ind, rna1.nor, width, color='#d62728')  # , yerr=menStd)
# p2 = plt.bar(ind, rna2.count_x, width, bottom=rna1.nor)  # , yerr=womenStd)



print(np.array(rna4.nor_y))

p1 = plt.bar(ind, rna4.nor_x, width, color='#d62728')  # , yerr=menStd)
# print(rna4.nor_y)
p2 = plt.bar(ind, rna4.nor_y, width, bottom=rna4.nor_x)  # , yerr=womenStd)
# p3 = plt.bar(ind, M, width, bottom=d)

plt.ylabel('Count')
plt.title('P. infestans')
# plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'G13'))
# plt.yticks(np.arange(0, 10000, 1000))
plt.legend((p1[0], p2[0]), ('Before ', 'After'))
plt.savefig('after and before.jpg',dpi=300)
plt.show()