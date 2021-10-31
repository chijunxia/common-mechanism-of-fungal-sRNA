FILENAME = "outer.csv"
outf= "sly_outer_aftermap_baiwan.csv"
stat = {}
out = open(outf, 'w')
with open(FILENAME,"r") as file:
    oldValue = 0
    sum=0
    for line in file:
        list_line = line.strip().split(',')
        val=int(list_line[1])
        #value = 1000000.00*int(list_line[1])*303592/(3864898*16553)#bc
        value = 1000000.00*int(list_line[1])/63026
        out.writelines(list_line[0]+","+str(value)+"\n")