import numpy as np
import matplotlib.pyplot as plt


# plot function

def plot(list1,list2,mode="default"):
    if mode=="default":
        plt.plot(list1,list2,"r")
        plt.xlabel("number of iterations")
        plt.ylabel("cost function values (J)")
        plt.show()
    elif mode=="scatter":
        plt.scatter(list1,list2)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()
    return

# Hypothesis

def Hypothesis(X,theta):
    result=np.dot(X,theta)
    return result
    

# cost function
def costFunction(hypothesis,Y,num_trainning):
#    print hypothesis.shape
#    print Y.shape
#    print np.subtract(hypothesis,Y).shape
#    print (np.subtract(hypothesis,Y)).shape

    #print np.dot(np.transpose(np.subtract(hypothesis,Y)),(np.subtract(hypothesis,Y))).shape
    cost_result = np.divide(np.dot(np.transpose(np.subtract(hypothesis,Y)),np.subtract(hypothesis,Y)),(2*num_trainning))
    return cost_result

def gradientDescent(num_iterations,alpha,num_trainning,X,Y,theta):
    
    Jvalues=np.zeros((1,num_iterations))
    
    for i in range(num_iterations):
        
        result1=np.dot(X.T,np.subtract(Hypothesis(X,theta),Y))*(alpha/num_trainning)
        theta=theta-result1
        
        Jvalues[0,i]=costFunction(Hypothesis(X,theta),Y,num_trainning)                                                                                                 
        
    return theta,Jvalues
    
   
   
   
#----------------------------------------- Code Start here ------------------------------------------------
# load data from file. here data consist of two features and the label.
data=np.genfromtxt("/home/waleed/PROJECTs/ML/DataSets/Ng_data/ex1data2.txt",
                  delimiter=",") 
      
# this for controlling the packing process of data to get x and y.
features_numbers=data.shape[1]-1
label=data.shape[1]-1 # just for clarity

# getting x and y , setting the algorithm values.          
x=data[:,0:features_numbers]
y=data[:,label]
y=np.reshape(data[:,1],(y.shape[0],1))  # just for reshape the y to be (n,1) instead of (n,).
m=y.size  # number of trainning set.
theta=np.zeros((x.shape[1],1))  
alpha=0.01 # learning rate
iterations=50



 #normalize x

mean_vector=np.zeros((x.shape[1],1))
std_vector=np.zeros((x.shape[1],1))
mean_vector[0,0]=np.mean(x[:,0])

std_vector[0,0]=np.std(x[:,0])
std_vector[1,0]=np.std(x[:,1])



x[:,0]=(x[:,0]-mean_vector[0,0])/std_vector[0,0]
x[:,1]=(x[:,1]-mean_vector[1,0])/std_vector[1,0]

# --------------------------------------- Cost function --------------------------------------------------
cost=costFunction(Hypothesis(x,theta),y,m)
# you can print the cost value befire gradient descent works
print cost
#-----------------------------------------Gradient descent -----------------------------------------------
new_theta,Jvalues=gradientDescent(num_iterations=iterations,alpha=alpha,num_trainning=m,X=x,Y=y,theta=theta)

# print gradient descent after working
print costFunction(Hypothesis(x,new_theta),y,m)

# ----------------------------------------Plotting --------------------------------------------------------

# i do  iterations_vector[0,:],Jvalues[0,:] instead of putting the two array directly because matplotlib 
#  works with lists not multi dimension .

iterations_vector=np.arange(1,iterations+1).reshape((1,iterations))
plot(iterations_vector[0,:],Jvalues[0,:])

