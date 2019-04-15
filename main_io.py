import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import csv
def accuracy(h,y):
    flag = 0
    n_flag=0
    for i in range(len(h)):
        vector = list()
        totalvector=list()
        #if z function return negative then it means sigmoid return 0
        #we can know that in order the plot of the sigmoid function
        if h[i] < 0.5:
            h[i] = 0
        elif h[i] >= 0.5:
            h[i] = 1
        #if it is matched the increase the rate
        if h[i] == y[i]:
            flag += 1
            vector.append(flag)

        #this is the missrate
        elif h[i] != y[i]:
            n_flag += 1

    plt.plot(array(vector))
    plt.ylabel("Accuracy")
    plt.show()
    return (flag / (flag + n_flag))*100


def LogisticRegression(batch_size:int,lr:float):

    f = open("ionosphere.data.csv", "r")
    reader = csv.reader(f, delimiter=",")
    dataset = np.asarray(list(reader))
    f.close()

    np.random.shuffle(dataset)
    split_by = int(len(dataset) * 0.2)
    test_data = dataset[0:split_by,0:34].astype('float')
    train_data = dataset[split_by:len(dataset)]
    labels = train_data[:, 34:]
    print(labels)
    testlabels = test_data[:, 34:]
    realLabels = list()
    for i in range(len(labels)):
        if labels[i] == 'b':
            realLabels.append(1)
        else:
            realLabels.append(0)
    print(realLabels)
    test = list()

    for i in range(len(testlabels)):
        if testlabels[i] == 'b':
            test.append(1)
        else:
            test.append(0)

        labelsArray = array(realLabels)
        testArray = array(test)

    features = train_data[:, 0:34].astype('float')
    print(len(features))
    beta = np.zeros(features.shape[1] + 1).astype('float')
    x = np.empty([batch_size, features.shape[1] + 1])
    x[:,0] = 1
    arr = list()
    print(len(features))
    for i in range(0, 10000):
        #optimize the minibatch
        counter=int(i%((len(features)/batch_size-1)))
        #take a substring o row
        x[:, 1:35] = features[int(counter * batch_size):int((counter + 1) * batch_size), 0:34]
        #take it also labels without int type it gives an error
        y = labelsArray[int(counter * batch_size): int((counter + 1) * batch_size)]
        #find the predictor(b0,b1,b2.....bn)
        z = np.dot(x, beta)
        #express the MODEL(--h--)
        h = 1 / (1 + np.exp(-z))
        #calculate the gradient for updating new beta
        gradient = np.dot(x.T, (h-y)) / y.size
        beta -= lr * gradient

        #ı have done this part for tracing the loss function changes
        if i % 100 == 0:
            z = np.dot(x, beta)
            h = 1 / (1 + np.exp(-z))
            loss = (y * np.log(h) + (1 - y) * np.log(1 - h)).mean()
            arr.append((loss))
            plt.plot(array(arr))
            plt.ylabel("Loss")
            plt.show()
            print("Loss",loss)

            #according to loss function -0.15 increaing so slowly
            if loss > -0.15:
                break



    ones=np.ones((len(test_data),1))
    test_data=np.concatenate((ones,test_data),axis=1)

    # calculating loss function
    z = np.dot(test_data, beta)
    h = 1 / (1 + np.exp(-z))
    #total loss function
    print("loss: ", (testArray * np.log(h) + (1 - testArray) * np.log(1 - h)).mean())
    # just setting counter for count our true and false predictions
    val = accuracy(h,testArray)
    print(val)

if __name__ == '__main__':
    LogisticRegression(10,0.01)