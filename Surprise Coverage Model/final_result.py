import numpy as np
import matplotlib.pyplot as plt
w = open('result/sa_conv12_cifar_fgsm_lsa.txt','r')
l = w.readlines()
a = [[0 for i in range(4)] for j in range(101)]
i = 0
x = []

for line in l:
    x.append(eval(line))
print(len(x))
count = 0
a = np.array(a,dtype = float)
x = np.array(x,dtype = float)
cc = 0
for i in range(0,len(x)):
    if (count==101 or count==0):
            if count==101:
                 cc=cc+1        
            count = 0
        
    else:
        #print(cc)
        #print("\n")
        a[count][cc]=(x[i]-x[i-count]) 
    #print("i:{0}  a[count][cc]{1} x[i]{2}  count{3} \n".format(i,a[count][cc],x[i],count))
    count = count + 1 

b = []
q = []
for i in range(1,101):
  b.append(i)
for i in range(1,101):
    
    v = []
    for j in range(len(a[i])):
        v.append(a[i][j])
    v = np.array(v,dtype = float)
    
    v = sorted(v)
    for j in range(len(v)):
        print(v[j])
        print("\n")
    q.append((v[1]+v[2])/2)
    #q.append(v[0])
    #print("4:{0} 5:{1}\n".format(a[i][4],a[i][5]))
    #print(q[i-1])
b = np.array(b,dtype = float)
print(len(b))
np.savetxt('final_result/sa_conv12_cifar_fgsm_lsa.txt',q[0:100],fmt='%.6f')
print(len(b))
#plt.figure(figsize=(10,6))
#plt.scatter(b,a[1:101],s = 5)
#plt.xlabel("Percentage of Adversarial")
#plt.ylabel("Increment compared to Original Data")
#plt.title('sa_lenet_mnist_add')
#plt.savefig('pic_result/sa_lenet_mnist_add.png')
