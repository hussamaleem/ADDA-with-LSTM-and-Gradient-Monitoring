import optuna
import torch
import model
import dataloader as loader
import gradient_monitoring
import train_test
import torch.nn as nn
import utils
from optuna.trial import TrialState
import pathlib
import os
import plotting
import torch.optim as optim
import yaml
from datetime import datetime
from tqdm import trange



class OptunaOptim():
    
    def __init__(self,
                 main_path,
                 src_dataset_num,
                 trg_dataset_num,
                 net,
                 window_type,
                 deci_mat_method,
                 gm_method,
                 initiator
                 ):
        
        self.main_path = main_path
        self.net = net
        self.src_dataset_num = src_dataset_num
        self.trg_dataset_num = trg_dataset_num
        self.time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.window_type = window_type
        self.deci_mat_method = deci_mat_method
        self.gm_method = gm_method
        self.initiator = initiator
        self.best_loss_list = []
        self.best_epoch_list = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    
    def get_config(self):
        
        with open(os.path.join(self.main_path, 'config.yaml')) as file:
            config = yaml.safe_load(file)
        self.optuna_config = config['Optuna']
        self.training_config = config['Training']
        self.data_config = config['Dataloader']
        self.plot_config = config['Plotting']

        
    def path_creator(self,trial_num):
        
        self.result_path = os.path.join(self.main_path,'00_Results', 
                                        self.net,
                                        self.src_dataset_num + '-' +
                                        self.trg_dataset_num , 
                                        self.window_type,
                                        self.time_stamp,
                                        'Num_Trial ' + str(trial_num))
        paths = []    
        result_excel = pathlib.Path(self.result_path).parent
        self.result_excel = os.path.join(result_excel, ('Detail Result Files'))
        
        self.model_saving_path = os.path.join(self.main_path,
                                              '00_Checkpoint', 
                                              self.net,
                                              self.src_dataset_num + '-' +
                                              self.trg_dataset_num,
                                              self.window_type,
                                              self.time_stamp , 
                                              'Num_trial' + str(trial_num))

        self.model_loader_path = os.path.join(self.main_path,
                                             'src_models',
                                             self.net,
                                             self.src_dataset_num,
                                             self.window_type,
                                             'Model.pt')

        
        self.loss_plot_path = os.path.join(self.result_path, 'Loss Plots')
        self.rul_plot_path = os.path.join(self.result_path, 'RUL Plots')
        paths.extend([self.result_path,self.result_excel,self.model_saving_path,
                      self.loss_plot_path,self.rul_plot_path])
    
        for i in paths:
            
            if not os.path.exists(i):
                
                os.makedirs(i)
        
        
    def create_parameters(self,trial):
        
        self.lr_config = self.optuna_config['learning_rate']
        self.out_feature_config = self.optuna_config['out_feature']
        self.learn_fact_config = self.optuna_config['learn_fact']
        self.batch_size_config = self.optuna_config['batch_size']
        self.momentum_config = self.optuna_config['momentum']
        '''
        self.lr = trial.suggest_categorical('lr', [0.01,0.001])
        
        self.out_features = trial.suggest_categorical('Out_features', [32,64])
        '''
        self.lr = trial.suggest_float('lr', 
                                      self.lr_config['low'],
                                      self.lr_config['high'], log=True)
        
        self.batch_size = trial.suggest_int('Batch_Size',
                                            self.batch_size_config['low'],
                                            self.batch_size_config['high'],
                                            self.batch_size_config['step'])
        
        self.out_features = trial.suggest_int('Out_Features', 
                                              self.out_feature_config['low'],
                                              self.out_feature_config['high'], 
                                              step = self.out_feature_config['step'])
        
        self.learn_fact = trial.suggest_float('Learning_Factor', 
                                              self.learn_fact_config['low'],
                                              self.learn_fact_config['high'], log=True)
        
        self.momentum = trial.suggest_float('Momentum',
                                            self.momentum_config['low'],
                                            self.momentum_config['high'], log=True) 
        
    def load_models(self):
        
        self.src_encoder, params = utils.model_loader(encoder=True, 
                                                      path=self.model_loader_path)
        self.bottleneck_dim, _ ,self.src_window_length = params 
        self.trg_encoder, _ = utils.model_loader(encoder=True, 
                                                  path=self.model_loader_path)
        self.rul_predictor = utils.model_loader(encoder=False, 
                                                path=self.model_loader_path)
        self.discriminator = model.Discriminator(in_features=self.bottleneck_dim, 
                                                 out_features=self.out_features
                                                 ).to(self.device)
        
        self.trg_len = self.data_config[self.net][self.trg_dataset_num]
        if self.window_type == 'Fixed_Length':
            self.trg_len = self.src_window_length
        
    def get_loaders(self):
        
        src_datasets, _ = loader.load_datasets(window_length=self.src_window_length, 
                                               dataset_num=self.src_dataset_num, 
                                               data_config=self.data_config, 
                                               main_path=self.main_path)
        src_train_dataset, _ = src_datasets
        
        trg_datasets, self.test_lengths = loader.load_datasets(window_length=self.trg_len,
                                                               dataset_num=self.trg_dataset_num, 
                                                               data_config=self.data_config, 
                                                               main_path=self.main_path)
        trg_train_dataset, trg_test_dataset = trg_datasets
        
        if len(src_train_dataset) == max(len(src_train_dataset), 
                                         len(trg_train_dataset)):
            
            trg_sampler = loader.RepeatSampler(trg_train_dataset, 
                                               src_train_dataset)
            src_sampler = None
        else:
            src_sampler = loader.RepeatSampler(src_train_dataset, 
                                               trg_train_dataset)
            trg_sampler = None
            
        self.src_train_loader, _ = loader.dataloaders(train_dataset=src_train_dataset, 
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      sampler=src_sampler) 
        
        self.trg_train_loader, self.trg_test_loader = loader.dataloaders(train_dataset=trg_train_dataset, 
                                                                         test_dataset=trg_test_dataset, 
                                                                         batch_size=self.batch_size,
                                                                         shuffle=True,
                                                                         sampler=trg_sampler) 

        
    def training_setup(self):
        
        self.optimizer_discr = optim.Adam(self.discriminator.parameters(), 
                                          lr=self.lr, 
                                          weight_decay=1e-3)
        self.optimizer_trg = optim.Adam(self.trg_encoder.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=1e-3)    
    
        self.criterion_discr = nn.CrossEntropyLoss()
        self.criterion_trg = nn.MSELoss()
        
        self.gm = gradient_monitoring.GradientMonitoring(trg_encoder=self.trg_encoder, 
                                                         learn_fact=self.learn_fact, 
                                                         momentum=self.momentum, 
                                                         mome_mat_initiator=self.initiator, 
                                                         deci_mat_method=self.deci_mat_method,
                                                         device=self.device)
        
        '''
        self.scheduler_discr = optim.lr_scheduler.ExponentialLR(self.optimizer_discr, 
                                                                gamma = 0.99)
        self.scheduler_trg = optim.lr_scheduler.ExponentialLR(self.optimizer_trg, 
                                                              gamma = 0.99)
        '''
        
    def objective(self,trial):  
        
        num = trial.number
        test_score_list = []
        discr_training_loss = []
        total_trg_encoder_loss = []
        lr_list = []
        training_epoch_list = []
        self.test_loss_list = []
        self.best_epoch = 0
        self.best_loss = 0
        
        self.get_config()
        self.path_creator(trial_num = num)
        self.create_parameters(trial)
        self.load_models()
        self.get_loaders()
        self.training_setup()
        
        train_test_class = train_test.train_test_loops(src_encoder=self.src_encoder, 
                                                       trg_encoder=self.trg_encoder, 
                                                       discr=self.discriminator, 
                                                       device=self.device, 
                                                       discr_optimizer=self.optimizer_discr, 
                                                       trg_optimizer=self.optimizer_trg, 
                                                       discr_criterion=self.criterion_discr, 
                                                       trg_criterion=self.criterion_trg, 
                                                       rul_predictor=self.rul_predictor,
                                                       gm = self.gm,
                                                       gm_method=self.gm_method
                                                       ) 
        
        plot = plotting.train_test_plots(dataset_num=self.trg_dataset_num, 
                                         result_path=self.result_path,
                                         test_lengths=self.test_lengths,
                                         eng_list=self.plot_config[self.trg_dataset_num],
                                         plot_path=self.loss_plot_path,
                                         rul_plot_path=self.rul_plot_path
                                         )
        
        print('..........Initiating Training Loop.......') 
        pbar = trange(self.training_config['epochs'])              
        
        for epoch in pbar:

            pbar.set_description(f'Epoch {epoch}')
            training_epoch_list.append(epoch)
            discr_loss, trg_encoder_loss = train_test_class.training_loop(src_loader=self.src_train_loader, 
                                                                          trg_loader=self.trg_train_loader, 
                                                                          batch_size=self.batch_size,
                                                                          epoch = epoch
                                                                          )
            
            discr_training_loss.append(discr_loss)
            total_trg_encoder_loss.append(trg_encoder_loss)
            pbar.set_postfix(Discr_Loss = discr_loss, 
                             Trg_Encoder_Loss = trg_encoder_loss)
            
            #self.scheduler_discr.step()
            #self.scheduler_trg.step()
            current_lr = self.optimizer_discr.param_groups[-1]['lr']               #(self.scheduler_discr.get_last_lr())
            lr_list.append(float(current_lr))                                       #current_lr[0]
            
            if (((epoch+1) > self.training_config['waiting_epochs']) and 
                ((epoch+1) % self.training_config['interval'] == 0)):
                
                print('\n.........Initializing Testing Loop........')
                
                test_pred , test_targ, test_loss = train_test_class.testing_loop(test_data=self.trg_test_loader)

                plot.testing_plots(preds=test_pred, 
                                   labels=test_targ, 
                                   n_epoch=epoch)
                
                test_score,eng_scores, pred_list,label_list = utils.score_cal(test_preds=test_pred,
                                                                              test_labels=test_targ,
                                                                              test_lengths=self.test_lengths,
                                                                              test_loss=test_loss,
                                                                              epoch=epoch,
                                                                              result_path=self.result_path,
                                                                              testing=False)
                pbar.set_postfix(Discr_loss=discr_loss, 
                                 Test_loss=test_loss,
                                 Test_score=test_score)
                
                self.test_loss_list.append(test_loss)
                test_score_list.append(test_score)
                
                if test_score <= min(test_score_list):
                    
                    utils.model_saver(path=self.model_saving_path,
                                      trg_model=self.trg_encoder,  
                                      discr_model=self.discriminator,
                                      num_trial=num,
                                      window_length=self.trg_len,
                                      batch_size=self.batch_size,
                                      lr=self.lr,
                                      bottleneck_dim=self.bottleneck_dim,
                                      out_feature=self.out_features)
                    
                    utils.accuracy_excel(test_score_list=eng_scores, 
                                   epoch=epoch, 
                                   test_loss=test_loss, 
                                   prediction_list=pred_list, 
                                   target_list=label_list, 
                                   sum_score=test_score, 
                                   result_path=self.result_path)
                    
                    self.best_epoch = epoch
                    self.best_loss = test_loss
                
                trial.report(test_score, epoch)
                if trial.should_prune():
                    
                    raise optuna.exceptions.TrialPruned()
        
        self.best_loss_list.append(self.best_loss)
        self.best_epoch_list.append(self.best_epoch)
        
        utils.learning_rate_excel(lr_list=lr_list, 
                                  epoch_list=training_epoch_list , 
                                  loss_list=discr_training_loss,
                                  result_path=self.result_path)
        
        plot.loss_plots(loss=discr_training_loss, train=True, name='Discriminator Loss')
        plot.loss_plots(loss=total_trg_encoder_loss, train=True, name='Target Encoder Loss')        
        plot.loss_plots(loss=self.test_loss_list, train=False, name='Testing')
        
        return min(test_score_list)
    
    def run_objective(self, n_trials, start_up_trials):
        
        sampler = optuna.samplers.TPESampler(n_startup_trials=start_up_trials, 
                                             constant_liar=True)
        
        self.study = optuna.create_study(direction='minimize',
                                         sampler=sampler)
        self.study.optimize(self.objective, 
                            n_trials=n_trials, 
                            gc_after_trial=True)
    
    
    def create_summary(self):
        
        fig = optuna.visualization.plot_param_importances(self.study)
        fig.write_image(os.path.join(self.result_excel,
                                     'Hyperparameter Importance.jpeg'))
        pruned_trials = self.study.get_trials(deepcopy=False, 
                                              states=[TrialState.PRUNED])
        complete_trials = self.study.get_trials(deepcopy=False, 
                                                states=[TrialState.COMPLETE])   
        print('\nStudy Summary')
        print(f'Number of finished trials: {len(self.study.trials):2}')
        print(f'Number of pruned trials: {len(pruned_trials):2}')
        print(f'Number of completed trials: {len(complete_trials):2}')
        best_trial = self.study.best_trial
        print("Best trial:")
        print("  Value: ", best_trial.value)
        print("  Params: ")
        results_df = self.study.trials_dataframe()
        utils.param_results(best_trial = best_trial, result_excel = self.result_excel)
        utils.results_dataframe(results_df = results_df, 
                                result_excel = self.result_excel, 
                                best_epoch = self.best_epoch_list,
                                best_loss = self.best_loss_list)
