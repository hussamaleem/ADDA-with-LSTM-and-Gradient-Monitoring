import matplotlib.pyplot as plt
import torch


class train_test_plots():
    
    def __init__(self,
                 dataset_num: str,
                 result_path,
                 test_lengths,
                 eng_list,
                 plot_path,
                 rul_plot_path
                 ):
        self.dataset_num = dataset_num
        self.result_path = result_path
        self.test_lengths = test_lengths
        self.eng_list = eng_list
        self.plot_path = plot_path
        self.rul_plot_path = rul_plot_path
            
            
    def loss_plots(self,loss,train,name):
        
        if train:
            name = name
        else:
            name = 'Testing Loss'
            
        plt.plot(loss, 'r', linewidth = 2.5,
                 label = "Loss")
        plt.grid()  
        plt.ylabel(f'{name}', size = 15)
        plt.xlabel('Epochs', size = 15)
        plt.savefig(self.plot_path + f'/{name}', 
                    bbox_inches = 'tight', dpi=150)
        plt.close()
        
    def testing_plots(self,preds,labels,n_epoch):
            
        preds = torch.split(preds,self.test_lengths)
        labels = torch.split(labels,self.test_lengths)
        for i in self.eng_list:
            
            plt.plot(preds[i], 'r', linewidth = 2.5, 
                     label = "Predicted RUL")
            plt.plot(labels[i], 'b', linewidth = 2.5, 
                     label = "Actual RUL")
            plt.legend(loc='upper right')
            plt.grid()  
            plt.ylabel('Remaining Useful Life', size = 15)
            plt.xlabel('Cycles', size = 15)
            plt.title(f'{self.dataset_num} Test Engine ID {i+1}', size = 20)
            plt.savefig(self.rul_plot_path + 
                        f'/Testing RUL plot, n_epoch {n_epoch}, Eng_id {i+1}', 
                        bbox_inches = 'tight', dpi=150)
            plt.close('all')
            plt.close()
    