# **Imports** #
import numpy as np 
import pandas as pd 
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
import argparse
warnings.simplefilter('error', RuntimeWarning)
cwd = os.getcwd()
cwd=cwd+"\\"
# **Neural Network**#

class Function:
    
    #input  : X shape : no_of_examples x input_dim x 1
    #output : Y,Z shape : no_of_examples x output_dim x 1
    
    @staticmethod
    def sigmoid(X,Y=None,derivative=False):
        
        if(derivative == False):
            Z=1 /(1 + np.exp(-1*X));
            return Z;
        else:
            return (Y)*(1-Y);     
    
    @staticmethod
    def softmax(X,Y=None,derivative=False):
        
        if(derivative == False):
            Z=np.exp(X-np.mean(X,axis=1,keepdims=True));
            return Z/np.sum(Z,axis=1,keepdims=True);
        else:
            dim = np.shape(Y)[-2];
            Z = np.transpose(Y,(0,2,1)) * (np.identity(dim) + ((np.ones((dim,dim))-2*np.identity(dim))*Y))
            return Z 


    @staticmethod    
    def cross_entropy_softmax(X,Y=None,derivative=False):
        
        if(derivative == False):
            Z=np.exp(X);
            return Z/np.sum(Z,axis=1,keepdims=True);
        else:
            return np.ones(np.shape(X));
            
    
    @staticmethod
    def tanh(X,Y=None,derivative=False):
        if(derivative==False):
            return np.tanh(X);
        else:
            return 1 - Y*Y ;
            
    
    @staticmethod       
    def sse(true,pred,derivative=False):
        
        assert np.shape(pred)==np.shape(true)
        
        if(derivative):
            return 2*(pred-true);
        else:
            return np.sum((pred-true)**2);
    
    @staticmethod
    def cross_entropy(true,pred,derivative=False):
        
        assert np.shape(pred)==np.shape(true)
        
        if(derivative):
            return 0 
        else:
            return np.sum(-true*np.log(pred));
    
    @staticmethod
    def dummy(true,pred,derivative=False):
        
        assert np.shape(pred)==np.shape(true)
        
        if(derivative):
            return  pred-true;
        else:
            return np.sum(-true*np.log(pred));
    
    @staticmethod
    def one_hot_encoder(Y,dim = 0):
        unq = np.unique(Y);
        if(np.size(unq) > dim):
            dim = np.size(unq);
        OH = np.zeros((np.size(Y),dim,1))
        for i in unq :
            OH[ (i==Y),(unq==i),0 ]= 1;
        return OH;



class NNLayer :

    def __init__(self,n_input,n_output,act_func,b=True): 
        
        #input size  x1 , output size  x2 , activation function, bias
        
        self.size = [n_input,n_output]
        
        # Weight filling

        if(weight_filler=='normal'):self.W    = np.random.normal(loc=0,scale=1,size=(n_output,n_input));
        elif(weight_filler=='rand'):self.W    = np.random.rand(n_output,n_input);        
        
        if(b==True):
            if(weight_filler=='normal'): self.B    = np.random.rand(n_output,1);
            elif(weight_filler=='rand'):self.B    = np.random.normal(loc=0,scale=1,size=(n_output,1));
        else:
            self.B = np.zeros(n_output,1);

        # Activation funtion     
        self.activation_function  = act_func;
        

    def  f_pass(self,X) :
        
        #input : X : no_of_samples x n_input x 1   
        #output: no_of_samples x n_output x 1
        assert np.size(X.shape)==3 
        assert np.shape(self.W)[1] == np.shape(X)[-2]
        
        return (self.activation_function(np.matmul(self.W,X)+self.B)); 
    
    def b_pass(self,grad_out,X,Y):
        
        #grad w.r.t output : no_of_samples x n_output x 1 
        #input : X         : no_of_samples x n_input x 1 
        #function output Y: no_of_samples x n_output x 1 
        #grad w.r.t input : no_of_samples x n_input x 1                 
        
        assert np.size(X.shape)==3
        assert np.size(Y.shape)==3

        n_samples = X.shape[-3];
        dim_in    = X.shape[-2]; 

        grad_l_a = self.activation_function((np.matmul(self.W,X)+self.B),Y,True); #local gradient w.r.t actvations

        assert (np.shape(grad_out)[-3] == np.shape(grad_l_a)[-3])and(np.shape(grad_out)[-2] == np.shape(grad_l_a)[-2]) 
        
        if(np.shape(grad_l_a)[-1]==1):

            grad_a = grad_out * grad_l_a  ;

        else :

            grad_a = np.matmul(grad_l_a,grad_out) ;

        grad_w = grad_a * X.reshape(-1,1,dim_in);
        grad_b = grad_a ;
        grad_in = np.matmul(np.transpose(self.W),grad_a);
        
        return [grad_in,grad_w,grad_b]


class FeedForwardNN: 
    
    def __init__(self,l_size,act_func,lossfunction):
        
        #Set Seed
        np.random.seed(1234)
        self.layers=[]; # list contaning all layer objects

        self.layers.append("InputLayer");

        self.loss_function = lossfunction;

        for layer in np.arange(1,np.size(l_size)):
            self.layers.append(NNLayer(l_size[layer-1],l_size[layer],act_func[layer]));

        self.annealing_accuracy = 0;
    
    def print_model(self):
        print("Network Architecture");
        print("layer 0 " + "InputLayer");
        for layer in np.arange(1,np.size(self.layers)):
            print("layer "+str(layer)+" " + str(self.layers[layer].activation_function))
        print("loss function : "+str(self.loss_function));

    def forward_propagation(self,X,return_H=False):
            
        Y=X;
        H=[];  #H : collection of function values at each layer

        if(return_H):
            H.append(Y);

        for layer in np.arange(1,np.size(self.layers)) :
    
            Y = self.layers[layer].f_pass(Y);
            
            if(return_H):
                H.append(Y);

        Y_pred = Y;
        
        if(return_H==False):
            return Y_pred;
        else:
            return H ;       
    

    
    def backward_propagation(self,X,Y_true):
        
        H   = self.forward_propagation(X,True);
        
        
        grad  = self.loss_function(Y_true,H[-1],True);
        
        grad_W =[];
        grad_B =[];

        assert np.shape(grad) == np.shape(Y_true),str(str(np.shape(grad))+str(np.shape(true)))
        
        for layer in np.flip(np.arange(1,np.size(self.layers)),axis=0):

           grad,t_w,t_b = self.layers[layer].b_pass(grad,H[layer-1],H[layer]) 
           
           grad_W.append(t_w);
           
           grad_B.append(t_b);
    

        return grad_W,grad_B;     
            
            
            
    def train_algo(self,train_X, train_Y, val_X, val_Y,
                   file_path=cwd,
                   opt = 'adam', 
                   adam_param=(0.9,0.99,10**-8),
                   momentum_param =0.9 ,
                   eta=0.001, 
                   batch_size = 20, 
                   max_epochs= 5 ,
                   annealing= False):
        
        #Shape Info
        #train_Y,Val_Y is one hot : no_of_samples x out_dim x 1
        #train_X,Val_X            : no_of_samples x in_dim x 1
        
        #Initalization
        
        beta_1 = adam_param[0];
        beta_2 = adam_param[1];
        epsilon = adam_param[2];
        m_W = [0]*np.size(self.layers);
        m_B = [0]*np.size(self.layers);
        v_W = [0]*np.size(self.layers);
        v_B = [0]*np.size(self.layers);
        gamma = momentum_param;
        no_train_samples = np.shape(train_X)[-3];
       
        #File to write logs
        file_log_train = open(file_path +'log_train.txt', 'a+')
        file_log_val   = open(file_path +'log_val.txt', 'a+')
        
        for epoch in np.arange(max_epochs):

            sample_id  = np.random.permutation(no_train_samples); #randomising batch
            
            if(annealing == True):
                Annealing_layers = self.layers;
            
            for step in np.arange(int(no_train_samples/batch_size)): # each epoch
                
                temp_X = train_X[sample_id[step*batch_size:(step+1)*batch_size],:,:];
                temp_Y = train_Y[sample_id[step*batch_size:(step+1)*batch_size],:,:]; 
                temp_W,temp_B =self.backward_propagation(temp_X,temp_Y); 
                
                #Asserting sizes of layers
                
                for layer in np.arange(1,np.size(self.layers)):
                    assert np.shape(self.layers[layer].W)==np.shape(np.sum(temp_W[-1*layer],axis=0))
                    
                if(opt=='gd'):
                    for layer in np.arange(1,np.size(self.layers)): 
                        self.layers[layer].W = self.layers[layer].W - eta * np.sum(temp_W[-1*layer],axis=0);
                        self.layers[layer].B = self.layers[layer].B - eta * np.sum(temp_B[-1*layer],axis=0);
                        
                elif(opt=='momentum'):
                    for layer in np.arange(1,np.size(self.layers)):
                        m_W[layer] = gamma * m_W[layer] + eta * np.sum(temp_W[-1*layer],axis=0);
                        m_B[layer] = gamma * m_B[layer] + eta * np.sum(temp_B[-1*layer],axis=0);
                        
                        self.layers[layer].W = self.layers[layer].W - m_W[layer];
                        self.layers[layer].B = self.layers[layer].B - m_B[layer];
                
                elif(opt=='nag'):
                    for layer in np.arange(1,np.size(self.layers)):
                        m_W[layer] = gamma * m_W[layer] + eta * np.sum(temp_W[-1*layer],axis=0);
                        m_B[layer] = gamma * m_B[layer] + eta * np.sum(temp_B[-1*layer],axis=0);

                        self.layers[layer].W = self.layers[layer].W - gamma * m_W[layer] - eta * np.sum(temp_W[-1*layer],axis=0);
                        self.layers[layer].B = self.layers[layer].B - gamma * m_B[layer] - eta * np.sum(temp_B[-1*layer],axis=0);

                elif(opt=='adam'):
                    for layer in np.arange(1,np.size(self.layers)): 
            
                        m_W[layer] = beta_1 * m_W[layer] + (1-beta_1) * np.sum(temp_W[-1*layer],axis=0);
                        m_B[layer] = beta_1 * m_B[layer] + (1-beta_1) * np.sum(temp_B[-1*layer],axis=0);
                        
                        v_W[layer] = beta_2 * v_W[layer] + (1-beta_2) * np.sum(temp_W[-1*layer],axis=0)* np.sum(temp_W[-1*layer],axis=0);
                        v_B[layer] = beta_2 * v_B[layer] + (1-beta_2) * np.sum(temp_B[-1*layer],axis=0)* np.sum(temp_B[-1*layer],axis=0);
                        
                        denom_W = (np.sqrt(v_W[layer]) + epsilon)*(1-beta_1) ;
                        denom_B = (np.sqrt(v_B[layer]) + epsilon)*(1-beta_1) ;
                        
                        num_W = eta*np.sqrt(1-beta_2)* m_W[layer];
                        num_B = eta*np.sqrt(1-beta_2)* m_B[layer];
                        
                        assert num_W.shape == self.layers[layer].W.shape
                        assert num_B.shape == self.layers[layer].B.shape
                        
                        self.layers[layer].W = self.layers[layer].W - num_W/denom_W ;
                        self.layers[layer].B = self.layers[layer].B - num_B/denom_B ;

                if((step%100==0)and(output_logs)):
                    log_Y_train = self.forward_propagation(train_X);
                    log_Y_val   = self.forward_propagation(val_X);
                    train_loss = self.loss_function(train_Y,log_Y_train);
                    train_error = 100-Metrics.accuracy(self.argmax(log_Y_train,axis=1),np.argmax(train_Y,axis=1));
                    val_loss = self.loss_function(val_Y,log_Y_val);
                    val_error = 100-Metrics.accuracy(self.argmax(log_Y_val,axis=1),np.argmax(val_Y,axis=1));
                    file_log_train.write("Epoch " +str(epoch)+", Step "+str(step)+" loss: "+ str(train_loss)+", Error: "+str(train_error)+", lr: "+str(eta)+"\n");
                    file_log_val.write("Epoch " +str(epoch)+", Step "+str(step)+" loss: "+ str(val_loss)+", Error: "+str(val_error)+", lr: "+str(eta)+"\n");
            

            # Terminal Logs
            
            current_epoch_accuracy_val = Metrics.accuracy(self.predict(val_X),np.argmax(val_Y,axis=1));
            if(show_output):
                current_epoch_output_train  = self.forward_propagation(train_X);
                current_epoch_loss_train  = self.loss_function(train_Y,current_epoch_output_train);
                current_epoch_accuracy_train = Metrics.accuracy(np.argmax(current_epoch_output_train,axis=1),np.argmax(train_Y,axis=1));
                print(epoch)
                print("Validation accuracy: " +str(current_epoch_accuracy_val));
                print("Train accuracy: " +str(current_epoch_accuracy_train));
                print("Train_loss:"+str(current_epoch_loss_train));
            
            #Annealing
            if(annealing == True):
                if(self.annealing_accuracy<current_epoch_accuracy_val):
                    self.annealing_accuracy = current_epoch_accuracy_val;
                else:
                    print("Annealing")
                    eta = eta / 2 ;
                    self.layers = Annealing_layers;

        file_log_train.close();
        file_log_val.close();        
        
    
    def predict(self,X):  #returns labels
        
        Y_pred = self.forward_propagation(X);
        
        return np.argmax(Y_pred,axis=1);  
    

# **save model and load model**#
def save_model(model,epoch):
        
        #writes model into file
        filename= IA.save_dir + "FNN_model_e:" + str(epoch) + ".pickle";
        filehandler = open(filename, 'wb') ;
        pickle.dump(model, filehandler);
        filehandler.close();
        
def load_model(epoch):

        #loades model from file
        filename= IA.save_dir + "FNN_model_e:" + str(epoch) + ".pickle";
        filehandler = open(filename, 'rb'); 
        Model = pickle.load(filehandler);
        filehandler.close();
        return Model
# **Metrics class**#

class Metrics:
    
    #Accuracy function
    @staticmethod
    def accuracy(true,predict):
        
        return np.mean(true==predict)*100;
    

# **Argument Praser** #

parser = argparse.ArgumentParser()
parser.add_argument("--lr", help="initial learning rate for gradient descent based algorithms", type = float)
parser.add_argument("--momentum", help="momentum to be used by momentum based algorithms", type = float)
parser.add_argument("--num_hidden", help = "number of hidden layers", type = int)
parser.add_argument("--sizes", help="a comma separated list for the size of each hidden layer", type = str)
parser.add_argument("--activation", help="the choice of activation function - valid values are tanh/sigmoid", type = str)
parser.add_argument("--loss", help="possible choices are squared error[sq] or cross entropy loss[ce]", type = str)
parser.add_argument("--opt", help="the optimization algorithm to be used: gd, momentum, nag, adam", type = str)
parser.add_argument("--batch_size", help="the batch size to be used - valid values are 1 and multiples of 5", type = int)
parser.add_argument("--epochs", help="number of passes over the data", type = int)
parser.add_argument("--anneal", help="if trur the algorithm should halve the learning rate if at any epoch the validation error increases and then restart that epoch", type = str)
parser.add_argument("--save_dir", help="the directory in which the pickled model should be saved", nargs='?', const=cwd,type = str)
parser.add_argument("--expt_dir", help="the directory in which the log files will be saved",nargs='?', const=cwd, type = str)
parser.add_argument("--train", help="path to the training dataset", type = str)
parser.add_argument("--val", help="path to the validation dataset", type = str)
parser.add_argument("--test", help="path to the test dataset", type = str)
parser.add_argument("--pretrain", help="Flag to use pre train model",nargs='?', const='False', type = str)
parser.add_argument("--state", help="path to the validation dataset",nargs='?', const= 0 ,type = int)
parser.add_argument("--testing", help="Flag to test model",nargs='?', const='True' ,type = str)
parser.add_argument("--logs", help="Flag to either print logs or not ",nargs='?', const='False' ,type = str)
IA = parser.parse_args()

print(IA);
# **Hyper Parameters** #

pca_com = 40 ;   # no of components to be selected  in PCA
if(IA.logs=='True'):output_logs = True ;          # flag to write logs in a file
else:output_logs = False;
show_output = True ;           # flag to to show outputs in terminal
weight_filler = 'rand';      # weight filler type

# **Preprocessing** #

# Importing data 
train_data = pd.read_csv(IA.train)
val_data = pd.read_csv(IA.val)
test_data = pd.read_csv(IA.test)


train_X = train_data.iloc[:,1:-1]
train_Y = train_data.iloc[:,-1]

val_X = val_data.iloc[:,1:-1]
val_Y = val_data.iloc[:,-1]

test_X = test_data.iloc[:,1:]

train_X = train_X.values
val_X = val_X.values
test_X = test_X.values

# Normalize input data
min_max_scalar = MinMaxScaler()
min_max_scalar.fit(train_X)

train_X = min_max_scalar.transform(train_X)
val_X = min_max_scalar.transform(val_X)
test_X = min_max_scalar.transform(test_X)

# Apply PCA for input data
pre_pca = PCA(n_components=pca_com)
pre_pca.fit(train_X);
train_X = pre_pca.transform(train_X); 
val_X   = pre_pca.transform(val_X);
test_X   = pre_pca.transform(test_X);

# Reshape input data
train_X = np.reshape(train_X,(-1,np.shape(train_X)[-1],1))
val_X = np.reshape(val_X,(-1,np.shape(val_X)[-1],1))
test_X = np.reshape(test_X,(-1,np.shape(test_X)[-1],1))

# Encode labels as one hot vector

train_Y = Function.one_hot_encoder(train_Y)
val_Y = Function.one_hot_encoder(val_Y)
print(np.shape(train_Y))
print(np.shape(val_Y))

#**Build Model**#
if(IA.pretrain =='True'):
    Model = load_model(int(IA.state));
else:
    IA_sizes = IA.sizes.strip('[').strip(']').split(',');
    l_sizes =[pca_com]+ [int(i) for i in IA_sizes] +[10];

    if(IA.activation == 'sigmoid') : l_acti  =["Input"] +[Function.sigmoid] * int(IA.num_hidden) ;
    elif(IA.activation == 'tanh')  : l_acti  =["Input"] +[Function.tanh]*int(IA.num_hidden) ;

    if(IA.loss == 'ce') : 
        l_acti = l_acti + [Function.cross_entropy_softmax];
        loss_func = Function.dummy;
    elif(IA.loss == 'sq'):
        l_acti = l_acti + [Function.softmax];
        loss_func = Function.sse;

    Model = FeedForwardNN(l_sizes,l_acti,loss_func);

# Train Model
if(IA.anneal=='True'):flag_annealing = True;
else :flag_annealing = False;
print(flag_annealing);
Model.print_model()
Model.train_algo(train_X,train_Y,val_X,val_Y,
                 file_path = IA.expt_dir,
                 max_epochs=IA.epochs,
                 annealing=flag_annealing,
                 opt=IA.opt,
                 batch_size=IA.batch_size,
                 momentum_param=IA.momentum,
                 eta=IA.lr)
#save model
save_model(Model,int(IA.state));
# Predicting Test
if(IA.testing=='True'):
    test_pred = Model.predict(test_X)
    sub = pd.DataFrame()
    sub['id'] = test_data['id']
    sub['label'] = test_pred.reshape(-1)
    sub.to_csv(IA.expt_dir+'test_submission.csv',index =False)
else:
    pass;