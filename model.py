import torch.nn as nn

class LSTMNetwork(nn.Module):
    
    def __init__(self,
                 hidden_size,
                 in_channel
                 ):
        super(LSTMNetwork,self).__init__()
        
        self.hidden_size= hidden_size
        self.lstm = nn.LSTM(in_channel, hidden_size, 
                            batch_first=True, 
                            bidirectional = True, 
                            num_layers=2,
                            )
        
    def forward(self,x):
        
        lstm_out , (self.hidden,self.cell) = self.lstm(x)
        return lstm_out[:,-1,:]
    
class LSTM_RUL_Predictor(nn.Module):
    def __init__(self,
                 hidden_size,
                 out_feature
                 ):
        
        super(LSTM_RUL_Predictor,self).__init__()
    
        self.linear_1 = nn.Linear(hidden_size*2, out_feature)
        self.linear_2 = nn.Linear(out_feature, 1)
        self.leakyrelu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self,src):

        out_1 = self.leakyrelu(self.linear_1(src))
        out_2 = self.dropout(out_1)
        out = self.linear_2(out_2)
        
        return out
    
class LSTM_Model(nn.Module):
    
    def __init__(self,hidden_size, out_feature, in_channel):
        
        super(LSTM_Model,self).__init__()
        
        self.lstm_1 = LSTMNetwork(hidden_size = hidden_size, in_channel = in_channel)
        self.lstm_rul_predictor = LSTM_RUL_Predictor(hidden_size = hidden_size, out_feature = out_feature)
        
    def forward(self,data):
        

        lstm_out_1 = self.lstm_1(data)
        out = self.lstm_rul_predictor(lstm_out_1)
        
        return out
    
    def weight_initialization(self,init_function):
        if init_function == 'xavier_n':
            initializer = nn.init.xavier_normal_
        elif init_function == 'kaiming_n':
            initializer = nn.init.kaiming_normal_
        elif init_function == 'xavier_u':
            initializer = nn.init.xavier_uniform_
        elif init_function == 'kaiming_u':
            initializer = nn.init.kaiming_uniform_
        for m in self.modules():
            if type(m) is nn.LSTM:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:       
                        initializer(param.data)          
                    if 'weight_hh' in name:          
                        initializer(param.data)          
            if type(m) is nn.Linear:
                for name, param in m.named_parameters():
                    if 'weight' in name:                         
                        initializer(param.data)
                        
                        
'-------------------------------------------------------------------------------------------------------------------------------'

class ConvolutionalNetwork(nn.Module):
    def __init__(self,
                 out_channel,
                 in_channel
                 ):
        
        super(ConvolutionalNetwork,self).__init__()
        self.layer_1 = nn.Sequential(nn.Conv1d(in_channel, out_channel, kernel_size = 7, stride = 1),
                                     nn.BatchNorm1d(out_channel),
                                     nn.LeakyReLU(),
                                     )
        
        self.layer_2 = nn.Sequential(nn.Conv1d(out_channel, (out_channel*3)//4, kernel_size =  5, stride = 1),
                                     nn.LeakyReLU(),
                                     )
        
        self.layer_3 = nn.Sequential(nn.Conv1d((out_channel*3)//4, out_channel//2, kernel_size = 3, stride = 1),
                                     nn.LeakyReLU(),
                                     )
        
        self.flatten = nn.Flatten()
        
    def forward(self,x):
        x = x.reshape(x.shape[0],x.shape[2],x.shape[1])
        out_1 = self.layer_1(x)
        out_2 = self.layer_2(out_1)
        out_3 = self.layer_3(out_2)
        #out_4 = self.layer_4(out_3)
        out = self.flatten(out_3)
        return  out
    
class CNN_RUL_Predictor(nn.Module):
    def __init__(self,
                 in_features,
                 out_features
                 ):
        
        super(CNN_RUL_Predictor,self).__init__()

        self.linear_1 = nn.Linear(in_features, out_features)
        self.linear_2 = nn.Linear(out_features, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self,src):
        
        out_1 = self.relu(self.linear_1(src))
        out_2 = self.dropout(out_1)
        out = (self.linear_2(out_2))
        
        return out
    
class CNN_Model(nn.Module):
    
    def __init__(self,out_features,out_channel,multiplier, in_channel):
        
        super(CNN_Model,self).__init__()
        

        self.cnn = ConvolutionalNetwork(out_channel = out_channel, in_channel = in_channel)
        self.rul_predictor = CNN_RUL_Predictor(in_features = (out_channel//2)*multiplier, out_features = out_features)
        
    def forward(self,data):

        out_1 = self.cnn(data)
        out = self.rul_predictor(out_1)
        
        return out
        
        
    def weight_initialization(self,init_function):
        if init_function == 'xavier_n':
            initializer = nn.init.xavier_normal_
        elif init_function == 'kaiming_n':
            initializer = nn.init.kaiming_normal_
        elif init_function == 'xavier_u':
            initializer = nn.init.xavier_uniform_
        elif init_function == 'kaiming_u':
            initializer = nn.init.kaiming_uniform_
        for m in self.modules():
            if type(m) is nn.Conv2d:
                for name, param in m.named_parameters():
                    if 'weight' in name:       
                        initializer(param.data)
            if type(m) is nn.Linear:
                for name, param in m.named_parameters():
                    if 'weight' in name:                        
                        initializer(param.data) 
'-------------------------------------------------------------------------------------------------------'

class Discriminator(nn.Module):
    def __init__(
            self,
            in_features,
            out_features
            ):
        
        super(Discriminator,self).__init__()
        
        self.linear_1 = nn.Linear(in_features,out_features)
        self.linear_2 = nn.Linear(out_features,out_features//2)
        self.linear_3 = nn.Linear(out_features//2,2)
        #self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU()
        
    def forward(self,features):
        
        out_1 = self.linear_1(features)
        out_2 = self.leakyrelu(out_1)
        out_3 = self.linear_2(out_2)
        out = self.linear_3(out_3)
        
        return out
    