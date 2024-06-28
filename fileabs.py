import os

if __name__=='__main__':
    path='F:/Robot/data_5'
    for root,dirs,files in os.walk(path):
        for file in files:
            if (not file.endswith('.txt') or "log" in file):
                continue
            # print(os.path.join(root,file))
            openfile=open(os.path.join(root,file),'r')
            labels=[]
            for line in openfile.readlines():
                label=line.split(' ')
                label=[abs(float(x)) for x in label]
                label[0]=int(label[0])
                labels.append(label)
            openfile.close()
            openfile=open(os.path.join(root,file),'w')
            for label in labels:
                openfile.write(' '.join([str(x) for x in label])+'\n')
            openfile.close()
            