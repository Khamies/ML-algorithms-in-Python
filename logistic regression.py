from __future__ import division
import numpy as np

import matplotlib.pyplot as plt 



# Hypothesis

def Hypothesis(X,theta):
    a=np.dot(X,theta)
    sigmoid=1/(1+np.exp(-a))
    return sigmoid
    

# cost function
def costFunction(hypothesis,Y,num_trainning):
    
    cost_result = -(1/num_trainning)*(np.dot(Y.T,np.log(hypothesis))+np.dot((1-Y).T,np.log(1-hypothesis)))
    return cost_result

def gradientDescent(num_iterations,alpha,num_trainning,X,Y,theta):
    
    Jvalues=np.zeros((1,num_iterations))
    
    for i in range(num_iterations):
        
        result1=np.dot(X.T,np.subtract(Hypothesis(X,theta),Y))*(alpha/num_trainning)
        theta=theta-result1
        
        Jvalues[0,i]=costFunction(Hypothesis(X,theta),Y,num_trainning)                                                                                                 
        
    return theta,Jvalues
    
    
def predict(theta, X):  
    probability = Hypothesis(X,theta)
    return [1 if x >= 0.5 else 0 for x in probability]    


#----------------------------------------- Code Start here ------------------------------------------------
# load data from file. here data consist of two features and the label.
data=np.genfromtxt("/home/waleed/PROJECTs/ML/DataSets/Ng_data/ex2data1.txt",
                  delimiter=",") 
          

# this for controlling the packing process of data to get x and y.
features_numbers=data.shape[1]-1
label=data.shape[1]-1 # just for clarity

# getting x and y , setting the algorithm values.          
x=data[:,0:features_numbers]
y=data[:,label]

y=np.reshape(data[:,label],(y.shape[0],1))  # just for reshape the y to be (n,1) instead of (n,).
m=y.size  # number of trainning set.
theta=np.zeros((x.shape[1],1))  
alpha=0.01 # learning rate
iterations=10000

#normalize x

mean_vector=np.zeros((x.shape[1],1))
std_vector=np.zeros((x.shape[1],1))

#normalize x
mean_vector=np.zeros((x.shape[1],1))
std_vector=np.zeros((x.shape[1],1))

for i in range(features_numbers):
    
    mean_vector[i,0]=np.mean(x[:,i])
    mean_vector[i,0]=np.mean(x[:,i])

    std_vector[i,0]=np.std(x[:,i])
    std_vector[i,0]=np.std(x[:,i])

x=np.divide(np.subtract(x,mean_vector.T),std_vector.T)


# --------------------------------------- Cost function --------------------------------------------------
cost=costFunction(Hypothesis(x,theta),y,m)
# you can print the cost value befire gradient descent works
print cost

  
new_theta,Jvalues=gradientDescent(num_iterations=iterations,alpha=alpha,num_trainning=m,X=x,Y=y,theta=theta)

# print gradient descent after working
print costFunction(Hypothesis(x,new_theta),y,m)
  

# calculating model accuracy
predictions = predict(new_theta, x)  
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]  
accuracy = (sum(map(int, correct)) % len(correct))  
print 'accuracy = {0}%'.format(accuracy)   