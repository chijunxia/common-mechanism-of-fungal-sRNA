sample_mfe = {}

import os

sample = 'D:\H\chi_paper_xin\Ziyouneng\sly.txt'
file = open(sample, "r")
outf= "D:\H\chi_paper_xin\Ziyouneng\sly_deal.txt"
out = open(outf, 'w')
count = 1

key = ''
val = 0
for line in file:
    if count % 3 == 2:
        key = line[0:len(line) - 1]
        #print(key)
    elif count % 3 == 0:
        val = line[-8:-2]
        sample_mfe[key] = float(val)
        print(sample_mfe[key])
        #print(val)
    count = count + 1
print(len(sample_mfe))

#for ii in sample_mfe.values():
#   print(ii)
out.write(str(sample_mfe))
