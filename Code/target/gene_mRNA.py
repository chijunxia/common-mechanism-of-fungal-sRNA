inf = "CDS_zong.txt"
outf= "CDS_mRNA_protin.txt"
def readwrite1(inf,outf):
    f=open(inf,'r')
    out=open(outf,'w')

    for line in f.readlines():

        line=line.replace(';','\t')
        list_line = line.strip().split()
        x = list_line[1] + "\t" + list_line[2]  + "\t"+list_line[3]+"\n"
        print(x)
        x=x.replace("ID=cds-",'')
        x=x.replace("Parent=rna-",'')
        x=x.replace("Dbxref=GeneID:",'')
        print(x)
        out.writelines(x)

    f.close()
    out.close()
readwrite1(inf,outf)