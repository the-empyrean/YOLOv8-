import os
import shutil
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
import random
def readLabel(file_id):
    f=open(file_id+'.txt','r')
    rst=[]
    for line in f.readlines():
        line=line.strip('\n')
        #按照空格划分
        line=line.split(' ')
        # print(line)
        rst.append([float(line[1]),float(line[2]),float(line[3]),float(line[4]),float(line[0])])
    f.close()
    return rst
def find_intersection(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2

    # 计算交集的左下角坐标
    left_bottom_x = max(x1, x3)
    left_bottom_y = max(y1, y3)

    # 计算交集的右上角坐标
    right_top_x = min(x2, x4)
    right_top_y = min(y2, y4)

    # 检查是否存在交集
    if right_top_x > left_bottom_x and right_top_y > left_bottom_y:
        return (left_bottom_x, left_bottom_y, right_top_x, right_top_y)
    else:
        return None


def ImgCuter(img_path,lab_path,img_target,lab_target,file):
    file_id = file[:-4]
    shutil.copy(img_path+file,img_target+file)
    shutil.copy(lab_path+file_id+'.txt',lab_target+file_id+'.txt')
    boxs=readLabel(lab_target+file_id)
    img0=cv2.imread(img_path+file)
    height,width=img0.shape[0],img0.shape[1]
    l=[((0,0),(height//2,width//2)),
        ((0,width//2),(height//2,width)),
        ((height//2,0),(height,width//2)),
        ((height//2,width//2),(height,width)),
        ((height//4,width//4),(height//4*3,width//4*3))]
    for i,rect in enumerate(l):
        yl,xl=rect[0]
        yr,xr=rect[1]
        img=img0[int(yl):int(yr),int(xl):int(xr)]
        content=''
        tag=0
        for box in boxs:
            x,y,w,h=box
            x1=int(x*width)-int(w*width/2)
            y1=int(y*height)-int(h*height/2)
            x2=int(x*width)+int(w*width/2)
            y2=int(y*height)+int(h*height/2)
            a=find_intersection((x1,y1,x2,y2),(xl,yl,xr,yr))
            if a!=None:
                tag=1
                x1,y1,x2,y2=a
                # cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                x1-=xl
                y1-=yl
                x2-=xl
                y2-=yl
                content+='0 %f %f %f %f'%((x1+x2)/2/(width/2),(y1+y2)/2/(height/2),
                    (x2-x1)/(width/2),(y2-y1)/(height/2))
        if (tag==1):
            out=open(lab_target+file_id+'_'+str(i)+'.txt','w')
            out.write(content)
            out.close()
            cv2.imwrite(img_target+file_id+'_'+str(i)+'.jpg',img)

def ImgCopy(img_path,lab_path,img_target,lab_target,file):
    file_id = file[:-4]
    shutil.copy(img_path+file,img_target+file)
    shutil.copy(lab_path+file_id+'.txt',lab_target+file_id+'.txt')

#直方图均衡化
def ImgHistEqualize(file,img_path,lab_path,img_target=None,lab_target=None):
    file_id = file[:-4]
    # shutil.copy(img_path+file,img_target+file)
    if (img_target!=None):
        shutil.copy(lab_path+file_id+'.txt',lab_target+file_id+'.txt')
    img0=cv2.imread(img_path+file)
    (b,g,r) = cv2.split(img0) #通道分解
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    result = cv2.merge((bH,gH,rH))#通道合成
    # res = np.hstack((img,result))
    # cv2.imwrite(img_target+file,img)
    cv2.imshow('img',img0)
    cv2.imshow('result',result)
    cv2.waitKey(0)

def labelbox2Rect(box):
    x,y,w,h,c=box
    x1=x-w/2
    y1=y-h/2
    x2=x+w/2
    y2=y+h/2
    return (x1,y1,x2,y2)

# def ObjectPaste(PasteImgpath,box,ObjImgpath):
#     x1,y1,x2,y2=labelbox2Rect(box)
#     ObjImg=Image.open(ObjImgpath)
#     ObjImg=ObjImg.crop((x1*ObjImg.size[0],y1*ObjImg.size[1],x2*ObjImg.size[0],y2*ObjImg.size[1]))
#     PasteImg=Image.open(PasteImgpath)
#     x1=int(x1*PasteImg.size[0])
#     y1=int(y1*PasteImg.size[1])
#     x2=int(x2*PasteImg.size[0])
#     y2=int(y2*PasteImg.size[1])
#     ObjImg=ObjImg.resize((x2-x1,y2-y1))
#     PasteImg.paste(ObjImg,(x1,y1,x2,y2))
#     return PasteImg

def ObjectPaste(PasteImg,box,ObjImg):
    box1=box.copy()
    #0.2~0.8
    box1[0]=random.uniform(0.4,0.6)
    box1[1]=random.uniform(0.4,0.6)
    x1,y1,x2,y2=labelbox2Rect(box1)
    x1=int(x1*PasteImg.size[0])
    y1=int(y1*PasteImg.size[1])
    x2=int(x2*PasteImg.size[0])
    y2=int(y2*PasteImg.size[1])
    ObjImg=ObjImg.resize((x2-x1,y2-y1))
    PasteImg.paste(ObjImg,(x1,y1,x2,y2))
    return box1,PasteImg


def ImgPasteObj(possible,file,Pastedpath,path,class_num=None,targetpath=None):
    pasteds=os.listdir(Pastedpath+'images/')
    file_id = file[:-4]
    boxs=readLabel(path+'labels/'+file_id)
    if (len(boxs)==0):
        return
    # max_box=boxs[0]
    # kkk=0
    # for (i,box) in enumerate(boxs):
    #     if (box[4]==class_num):
    #         max_box=box
    #         kkk=1
    #         break
    # if (kkk==0):
    #     return
    for max_box in boxs:
        if (max_box[4]!=class_num):
            continue
        img0=Image.open(path+'images/'+file)
        x1,y1,x2,y2=labelbox2Rect(max_box)
        img0=img0.crop((x1*img0.size[0],y1*img0.size[1],x2*img0.size[0],y2*img0.size[1]))
        theta=90
        Rimg90=img0.rotate(theta,expand=True)
        Rimg180=img0.rotate(180,expand=True)
        Rimg270=img0.rotate(270,expand=True)
        k=0.1/max(max_box[2],max_box[3])

        max_box[2]*=float(k)
        max_box[3]*=float(k)
        # Rimg0.show()
        Rbox=[max_box[0],max_box[1],max_box[3],max_box[2],max_box[4]]
        for pasted in pasteds:
            if (random.random()<possible):
                pasteimg=Image.open(Pastedpath+'images/'+pasted)
                if (random.random()<0.25):
                    box,img=ObjectPaste(pasteimg,Rbox,Rimg90)
                elif (random.random()<0.5):
                    box,img=ObjectPaste(pasteimg,max_box,img0)
                elif (random.random()<0.75):
                    box,img=ObjectPaste(pasteimg,Rbox,Rimg180)
                else:
                    box,img=ObjectPaste(pasteimg,Rbox,Rimg270)
                img.save(targetpath+'images/'+file_id+'_'+pasted)
                pasted_id=pasted[:-4]
                shutil.copy(Pastedpath+'labels/'+pasted_id+'.txt',targetpath+'labels/'+file_id+'_'+pasted_id+'.txt')
                #在文件后添加一行
                out=open(targetpath+'labels/'+file_id+'_'+pasted_id+'.txt','a')
                out.write('%d %f %f %f %f'%(box[4],box[0],box[1],box[2],box[3]))

def pca_color_augmentation_modify(image_array_input):
    assert image_array_input.ndim == 3 and image_array_input.shape[2] == 3
    assert image_array_input.dtype == np.uint8
    img = image_array_input.reshape(-1, 3).astype(np.float32)
    ch_var = np.var(img, axis=0)
    scaling_factor = np.sqrt(3.0 / sum(ch_var))
    img = (img - np.mean(img, axis=0)) * scaling_factor

    cov = np.cov(img, rowvar=False)
    lambd_eigen_value, p_eigen_vector = np.linalg.eig(cov)

    rand = np.random.randn(3) * 0.1
#     rand = 0.3
    delta = np.dot(p_eigen_vector, rand*lambd_eigen_value)
    delta = (delta * 255.0).astype(np.int32)[np.newaxis, np.newaxis, :]

    img_out = np.clip(image_array_input + delta, 0, 255).astype(np.uint8)
    return img_out

def ImgColorAug(file,img_path,lab_path,img_target=None,lab_target=None):
    file_id = file[:-4]
    # shutil.copy(img_path+file,img_target+file)
    # if (img_target!=None):
    #     shutil.copy(lab_path+file_id+'.txt',lab_target+file_id+'.txt')
    img0=Image.open(img_path+file)
    img=img0.copy()
    if (img0.size[0]*img0.size[1]>=2073600):
        k=640/img0.size[0]
        img=img.resize((640,int(img.size[1]*k)))
    img0= np.array(img)
    # img0 = pca_color_augmentation_modify(img0)
    for i in range(0,2):
        img0 = pca_color_augmentation_modify(img0)
        img = Image.fromarray(img0)
        print(img.size[0],img.size[1])
        img.save(img_target+file_id+'_pca'+str(i)+'.jpg')
        shutil.copy(lab_path+file_id+'.txt',lab_target+file_id+'_pca'+str(i)+'.txt')

def ImgRotate(file,theta,img_path,lab_path,img_target=None,lab_target=None):
    img=Image.open(img_path+file)
    img=img.rotate(theta,expand=True) #theta单位为度，逆时针旋转
    img.save(img_target+'rotate_'+file)

def makeFinalData(path,targetpath):
    files=os.listdir(path+'images/')
    # print(path+'images/train')
    baggingpath=targetpath+'data_bagging/'
    os.makedirs(baggingpath+'pothole', exist_ok=True)
    os.makedirs(baggingpath+'normal', exist_ok=True)
    os.makedirs(targetpath+'images/train', exist_ok=True)
    os.makedirs(targetpath+'labels/train', exist_ok=True)
    os.makedirs(targetpath+'images/val', exist_ok=True)
    os.makedirs(targetpath+'labels/val', exist_ok=True)
    for (i,file) in enumerate(files[:]):
        print(file)
        file_id = file[:-4]
        if (random.random()<0.8):
            ImgCopy(path+'images/',path+'labels/',
                    targetpath+'images/train/',targetpath+'labels/train/',file)
        else:
            if (random.random()<0.5):
                ImgCopy(path+'images/',path+'labels/',
                    targetpath+'images/val/',targetpath+'labels/val/',file)
            else:
                boxs=readLabel(path+'labels/'+file_id)
                if (len(boxs)>0):
                    shutil.copy(path+'images/'+file,baggingpath+'pothole/'+file)
                else:
                    shutil.copy(path+'images/'+file,baggingpath+'normal/'+file)
#椒盐噪声
def SaltAndPepper(img_path,lab_path,img_target,lab_target,file):
    file_id = file[:-4]
    # shutil.copy(img_path+file,img_target+file)
    # shutil.copy(lab_path+file_id+'.txt',lab_target+file_id+'.txt')
    img0=cv2.imread(img_path+file)
    img=img0.copy()
    if (img0.shape[0]*img0.shape[1]>=2073600):
        k=640/img0.shape[0]
        img=cv2.resize(img,(int(img.shape[1]*k),640))
    height,width=img.shape[0],img.shape[1]
    cnt=int(height*width*0.005)
    for i in range(0,int(cnt)):
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    for i in range(0,cnt):
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 0
    cv2.imwrite(img_target+'SP_'+file,img)
    shutil.copy(lab_path+file_id+'.txt',lab_target+'SP_'+file_id+'.txt')

#高斯噪声
def GaussianNoise(img_path,lab_path,img_target,lab_target,file):
    file_id = file[:-4]
    # shutil.copy(img_path+file,img_target+file)
    # shutil.copy(lab_path+file_id+'.txt',lab_target+file_id+'.txt')
    img0=cv2.imread(img_path+file)
    img=img0.copy()
    if (img0.shape[0]*img0.shape[1]>=2073600):
        k=640/img0.shape[0]
        img=cv2.resize(img,(int(img.shape[1]*k),640))
    height,width=img.shape[0],img.shape[1]
    for i in range(0,height):
        for j in range(0,width):
            if (5<img[i,j,0] and img[i,j,0]<250): img[i,j,0]+=random.gauss(0,1)*5
            if (5<img[i,j,1] and img[i,j,1]<250): img[i,j,1]+=random.gauss(0,1)*5
            if (5<img[i,j,2] and img[i,j,2]<250): img[i,j,2]+=random.gauss(0,1)*5
    cv2.imwrite(img_target+'GN_'+file,img)
    shutil.copy(lab_path+file_id+'.txt',lab_target+'GN_'+file_id+'.txt')

#模糊
def Blur(img_path,lab_path,img_target,lab_target,file):
    file_id = file[:-4]
    # shutil.copy(img_path+file,img_target+file)
    # shutil.copy(lab_path+file_id+'.txt',lab_target+file_id+'.txt')
    img0=cv2.imread(img_path+file)
    img=img0.copy()
    if (img0.shape[0]*img0.shape[1]>=2073600):
        k=640/img0.shape[0]
        img=cv2.resize(img,(int(img.shape[1]*k),640))
    img=cv2.blur(img,(5,5))
    cv2.imwrite(img_target+'BL_'+file,img)
    shutil.copy(lab_path+file_id+'.txt',lab_target+'BL_'+file_id+'.txt')

def ImgRotate(file,theta,img_path,lab_path,img_target=None,lab_target=None):
    img=Image.open(img_path+file)
    print(img.size[0],img.size[1])
    # if (img.size[0]*img.size[1]>=2073600):
    #     k=640/img.size[0]
    #     img=img.resize((640,int(img.size[1]*k)))
    print(img.size[0],img.size[1])
    img=img.rotate(-theta,expand=True) #theta单位为度，逆时针旋转
    labels=readLabel(lab_path+file[:-4])
    # print(labels)
    for i in range(len(labels)):
        x,y,w,h,c=labels[i]
        x_r=(x-0.5)*np.cos(theta/180*np.pi)-(y-0.5)*np.sin(theta/180*np.pi)+0.5
        y_r=(x-0.5)*np.sin(theta/180*np.pi)+(y-0.5)*np.cos(theta/180*np.pi)+0.5
        w_r=abs(w*np.cos(theta/180*np.pi)+h*np.sin(theta/180*np.pi))
        h_r=abs(h*np.cos(theta/180*np.pi)-w*np.sin(theta/180*np.pi))
        labels[i]=(labels[i][4],x_r,y_r,w_r,h_r)
    # print(labels) 
    img.save(img_target+'rotate%d_'%(theta)+file)
    out=open(lab_target+'rotate%d_'%(theta)+file[:-4]+'.txt','w')
    for label in labels:
        out.write(str(int(label[0]))+' '+' '.join([str(x) for x in label[1:]])+'\n')


#main
if __name__ == '__main__':
    # path='./apple3/val/'
    # targetpath='./data_aug_apple/val/'

    # if not os.path.exists(targetpath): 
    #     os.makedirs(targetpath)
    # if not os.path.exists(targetpath+'images/'):
    #     os.makedirs(targetpath+'images/')
    # if not os.path.exists(targetpath+'labels/'):
    #     os.makedirs(targetpath+'labels/')
    # filelist=os.listdir(path+'images/')
    # paste_path='./data_1/train/'
    # for file in filelist:
    #     ImgPasteObj(0.001,file,paste_path,path,12,targetpath)

    path='./hanger3/test/'
    targetpath='./data_aug_hanger3/test/'
    if not os.path.exists(targetpath): 
        os.makedirs(targetpath)
    if not os.path.exists(targetpath+'images/'):
        os.makedirs(targetpath+'images/')
    if not os.path.exists(targetpath+'labels/'):
        os.makedirs(targetpath+'labels/')
    filelist=os.listdir(path+'images/')

    # for file in filelist:
    #     suf=file[-8:-4]
    #     if (suf[3]=='0'):
    #         ImgRotate(file,90,path+'images/',path+'labels/',targetpath+'images/',targetpath+'labels/')
    #         ImgRotate(file,180,path+'images/',path+'labels/',targetpath+'images/',targetpath+'labels/')
    #         ImgRotate(file,270,path+'images/',path+'labels/',targetpath+'images/',targetpath+'labels/')
    #     if (suf[2]=='1' or suf[1]=='1'):
    #         SaltAndPepper(path+'images/',path+'labels/',targetpath+'images/',targetpath+'labels/',file)
    #         GaussianNoise(path+'images/',path+'labels/',targetpath+'images/',targetpath+'labels/',file)
    #         Blur(path+'images/',path+'labels/',targetpath+'images/',targetpath+'labels/',file)
    #         ImgColorAug(file,path+'images/',path+'labels/',targetpath+'images/',targetpath+'labels/')
    for file in filelist:
        # suf=file[-8:-4]
        print(file)
        ImgRotate(file,90,path+'images/',path+'labels/',targetpath+'images/',targetpath+'labels/')
        ImgRotate(file,180,path+'images/',path+'labels/',targetpath+'images/',targetpath+'labels/')
        ImgRotate(file,270,path+'images/',path+'labels/',targetpath+'images/',targetpath+'labels/')
        if (random.random()<1): 
            SaltAndPepper(path+'images/',path+'labels/',targetpath+'images/',targetpath+'labels/',file)
        if (random.random()<1):    
            GaussianNoise(path+'images/',path+'labels/',targetpath+'images/',targetpath+'labels/',file)
        if (random.random()<1):    
            Blur(path+'images/',path+'labels/',targetpath+'images/',targetpath+'labels/',file)
        if (random.random()<1):
            ImgColorAug(file,path+'images/',path+'labels/',targetpath+'images/',targetpath+'labels/')        

    

    # for file in filelist:
    #     if random.random()<0.1:
    #         ImgRotate(file,90,path+'images/',path+'labels/',targetpath+'images/',targetpath+'labels/')
    #         ImgRotate(file,180,path+'images/',path+'labels/',targetpath+'images/',targetpath+'labels/')
    #         ImgRotate(file,270,path+'images/',path+'labels/',targetpath+'images/',targetpath+'labels/')

    filelist1=os.listdir(targetpath+'images/')
    if not os.path.exists(targetpath+'img_lab/'):
        os.makedirs(targetpath+'img_lab/')
    for file in filelist1:
        # if random.random()>0.1:
        #     continue
        img=Image.open(targetpath+'images/'+file)
        labs=readLabel(targetpath+'labels/'+file[:-4])
        for i in range(len(labs)):
            x,y,w,h,c=labs[i]
            draw=ImageDraw.Draw(img)
            draw.rectangle(((x-w/2)*img.size[0],(y-h/2)*img.size[1],(x+w/2)*img.size[0],(y+h/2)*img.size[1]),outline='red')
        img.save(targetpath+'img_lab/'+file)

    