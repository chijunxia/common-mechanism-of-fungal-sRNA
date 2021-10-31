inf1 = "sly_inner_8477.csv"
inf2= "bc_innner_8477.csv"
outf1="sly_grow.csv"
outf2="bc_grow.csv"
with open(inf1, "r") as f1:
    lineAs = f1.readlines()
with open(inf2, "r") as f2:
    lineBs = f2.readlines()
out1=open(outf1,'w')
out2=open(outf2,'w')
for lineA in lineAs:
    for lineB in lineBs:
        list_line=lineA.strip().split(',')
        lineF0=list_line[0]+'\n'
        lineF1=float(list_line[1])
        list_line1 = lineB.strip().split(',')
        lineC0 = list_line1[0] + '\n'
        lineC1 = float(list_line1[1])
        if lineC0 == lineF0:
           # Growth_rate=(lineF1-lineC1)/lineC1
            #if Growth_rate >= 2:
            if lineF1 > lineC1:
                out1.write(lineA)
                out2.write(lineB)
out1.close()
out2.close()