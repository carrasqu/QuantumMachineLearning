import numpy as np


muv=["0.00500", "0.00545", "0.00590", "0.00635", "0.00680", "0.00725", "0.00770", "0.00815", "0.00860", "0.00905", "0.00950", "0.00995", "0.01040", "0.01085", "0.01130", "0.01175", "0.01220", "0.01265", "0.01310", "0.01355", "0.01400", "0.01445", "0.01490", "0.01535", "0.01580", "0.01625", "0.01670", "0.01715", "0.01760", "0.01805", "0.01850", "0.01895", "0.01940", "0.01985", "0.02030", "0.02075", "0.02120", "0.02165", "0.02210", "0.02255", "0.02300", "0.02345", "0.02390", "0.02435", "0.02480", "0.02525", "0.02570", "0.02615", "0.02660", "0.02705", "0.02750", "0.02795", "0.02840", "0.02885", "0.02930", "0.02975", "0.03020", "0.03065", "0.03110", "0.03155", "0.03200", "0.03245", "0.03290", "0.03335", "0.03380", "0.03425", "0.03470", "0.03515", "0.03560", "0.03605", "0.03650", "0.03695", "0.03740", "0.03785", "0.03830", "0.03875", "0.03920", "0.03965", "0.04010", "0.04055", "0.04100", "0.04145", "0.04190", "0.04235", "0.04280", "0.04325", "0.04370", "0.04415", "0.04460", "0.04505", "0.04550", "0.04595", "0.04640", "0.04685", "0.04730", "0.04775", "0.04820", "0.04865", "0.04910", "0.04955", "0.05000", "0.05045", "0.05090", "0.05135"]

muv=["0.00500", "0.00545", "0.00590", "0.00635", "0.00680", "0.00725", "0.00770", "0.00815", "0.00860", "0.00905", "0.00950", "0.00995", "0.01040", "0.01085", "0.01130", "0.01175", "0.01220", "0.01265", "0.01310", "0.01355", "0.01400", "0.01445", "0.01490", "0.01535", "0.01580", "0.01625", "0.01670", "0.01715", "0.01760", "0.01805", "0.01850", "0.01895", "0.01940", "0.01985", "0.02030", "0.02075", "0.02120", "0.02165", "0.02210", "0.02255", "0.02300", "0.02345" ] 


L=200
lx=4
V=lx*lx*lx

V4d=V*L

testpercentage=20

#xx=np.genfromtxt('N4x4x4_L200_U9_Mu0_dtau0.00500.HSF.stream',dtype=np.int,delimiter=1, usecols =tuple(range(V4d)))

Tc=0.35

ytrain=np.asarray([],dtype=np.int8)
ytest=np.asarray([],dtype=np.int8)


k=0
o=0
do=0
for i in muv:
    
    fname='N4x4x4_L200_U9_Mu0_dtau'+i+'.HSF.stream'
    T=1.0/(L*float(i)) 
    xx=np.genfromtxt(fname,dtype=np.uint8,delimiter=1, usecols =tuple(range(V4d)))
    print fname, xx.shape, T
    
    ntest=int(xx.shape[0]*testpercentage/100)
    ntrain=xx.shape[0]-ntest
     
    train=xx[0:ntrain,:]
    test=xx[ntrain:,:]   
    if k==0:
      Xtrain=np.copy(train)
      Xtest=np.copy(test)
      k+=1
    else:  
      Xtrain=np.append(Xtrain,train, axis=0)
      Xtest=np.append(Xtest,test,axis=0) 
        
     
    if T<Tc:
       ytest=np.append(ytest,np.zeros(ntest))
       ytrain=np.append(ytrain,np.zeros(ntrain))
       o=o+1  
    else:
       ytest=np.append(ytest,np.ones(ntest))
       ytrain=np.append(ytrain,np.ones(ntrain))
       do=do+1
 
        
       
print 'do,o,ntest*do,ntest*o,ntrain*do,ntrain*o',do,o,ntest*do,ntest*o,ntrain*do,ntrain*o         
    
np.savetxt('ytest.txt',ytest,fmt='%1.1d')
np.savetxt('Xtest.txt',Xtest,fmt='%1.1d')

np.savetxt('ytrain.txt',ytrain,fmt='%1.1d')
np.savetxt('Xtrain.txt',Xtrain,fmt='%1.1d')
