import torch


class train_test_loops():
    def __init__(self,
                 src_encoder,
                 trg_encoder,
                 discr,
                 device,
                 discr_optimizer,
                 trg_optimizer,
                 discr_criterion,
                 trg_criterion,
                 rul_predictor,
                 gm,
                 gm_method
                 ):

        self.src_encoder = src_encoder
        self.trg_encoder = trg_encoder
        self.discr = discr
        self.device = device
        self.discr_optimizer = discr_optimizer
        self.trg_optimizer = trg_optimizer
        self.discr_criterion = discr_criterion
        self.trg_criterion = trg_criterion
        self.rul_predictor = rul_predictor
        self.gm = gm
        self.gm_method=gm_method
        
    def training_loop(self, 
                      src_loader,
                      trg_loader,
                      batch_size,
                      epoch):
        
        discr_train_loss = 0
        trg_encoder_loss = 0
        total_iterations = 0
        
        self.discr.train()
        self.trg_encoder.train()
        
        parameters = list(self.src_encoder.parameters()) + list(self.rul_predictor.parameters())
        for param in parameters:
            param.requires_grad = False
            
        for i,(src_loader, trg_loader) in enumerate(zip(src_loader, trg_loader)):
            self.discr_optimizer.zero_grad()
            self.trg_optimizer.zero_grad()
            
            src_data, _ = src_loader
            trg_data, _ = trg_loader
            
            src_data = src_data.to(self.device)
            trg_data = trg_data.to(self.device)
            
            src_feat = self.src_encoder(src_data)
            trg_feat = self.trg_encoder(trg_data)
            concat_feat = torch.cat((src_feat, trg_feat), dim = 0)
            
            src_labels = torch.ones((src_feat.size(0))).long().to(self.device)
            trg_labels = torch.zeros((trg_feat.size(0))).long().to(self.device)
            inverted_labels = torch.ones((trg_feat.size(0))).long().to(self.device)
            concat_labels = torch.cat((src_labels,trg_labels), dim = 0)
            
            discr_pred = self.discr(concat_feat)
            discr_loss = self.discr_criterion(discr_pred,concat_labels)
            discr_loss.backward()
            
            self.discr_optimizer.step()
            
            self.discr_optimizer.zero_grad()
            self.trg_optimizer.zero_grad()
            
            trg_feat = self.trg_encoder(trg_data)
            trg_pred = self.discr(trg_feat)
            trg_loss = self.discr_criterion(trg_pred, inverted_labels)
            trg_loss.backward()
            
            self.gm.method_selector(gm_method=self.gm_method,epoch=epoch if 
                                    self.gm_method == 'vgm' else None)
            
            self.trg_optimizer.step()
            
            discr_train_loss += discr_loss.item()
            trg_encoder_loss += trg_loss.item()
            total_iterations += 1

        total_discr_loss = discr_train_loss/total_iterations
        total_trg_loss = trg_encoder_loss/total_iterations
            
        return total_discr_loss, total_trg_loss

    def testing_loop(self, test_data):
        
        self.trg_encoder.eval()
        self.rul_predictor.eval()
        with torch.no_grad():
            
            test_loss = 0
            total_iterations = 0
            for i, (data, label) in enumerate(test_data):
                
                data, label = data.to(self.device), label.to(self.device)
                enc_pred = self.trg_encoder(data)
                pred = self.rul_predictor(enc_pred)
                loss = torch.sqrt(self.trg_criterion(pred, label))
                pred = pred.detach().cpu()
                label = label.detach().cpu()
                test_loss+=loss.detach().item()
                if i == 0:  
                    
                    predictions = pred
                    targets = label
                else:
                    
                    predictions = torch.cat((predictions, pred), dim = 0)
                    targets = torch.cat((targets, label), dim = 0)
                total_iterations += 1
            total_test_loss = test_loss/total_iterations
            
            return predictions, targets, total_test_loss


