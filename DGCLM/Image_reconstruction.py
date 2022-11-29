import cv2
import numpy as np

label = np.loadtxt("cluster_output.txt")
print(label.shape)

index = 0

for i in range(1,28):
    img = np.ones((256,256,3),np.uint8)
    data = np.loadtxt("./data/dicom/"+str(i)+".txt")
    
    for j in range(256):
        for k in range(256):
             for x in range(3):
                 img[j][k][x]=255
    for j in range(256):
        for k in range(256):
            if data[j*256+k][0]!=0:
                if label[index] == 0:
                    cv2.circle(img,(k,j),1,(0,0,255))
                elif label[index] == 1:
                    cv2.circle(img,(k,j),1,(255,0,0))
                index = index+1
                print(index)

             


    cv2.imwrite("./cluster_image/"+str(i)+"-cluster.png", img)
