import pandas as pd
#qinran1_1825_sort_len_count.csv
#3_4_qinran1_1825_sort_len_count.csv
#dwj1_sort_len1825_count.txt
path1=input("Please input the file to deal with:")
path2=input("Please input the file to output:")

data1 = pd.read_csv(path1)
data1.columns=['name','len','count']
ls2=data1['count'].values
lines=len(ls2)
i=lines*3/4-1
data=ls2[int(i)]
ls3=[]

j=0
for j in range(0,int(lines)):
    ls3.append(ls2[j]/data)
data11=data1.T
data21=[ls3]
data2=data11.append(data21)
data2.index=['name','len','count','nor']
data=data2.T
#print(data)
data.to_csv(path2, index = None)
# file1=open(path1,"r")
# lines=len(file1.readlines())
# print(lines)
# i=1
# ls=[]
# for a in file1:
#     print(a)
#     #if i==((lines-1)*3/4)+1:
#     #    ls.append(a)
#     #    print(a)
#     #    break
#     i=i+1

#count0=ls[2]
#print(count0)
#print(ls)