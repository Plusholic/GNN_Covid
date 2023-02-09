# from stgraph_trainer.trainers.base import BaseTrainer
import torch

class BaseTrainer():
  """
  The Trainer base class containing all information for training and predicting.

  Parameters
  ----------
  model: object
    The model to train and predict.

  train_ds: object
    The train dataset.

  test_ds: object
    The test dataset.

  loss_func: object
    The loss object function.

  optimizer: object
    The optimizer used to train the model.

  callbacks: list or objects, default None
    The list of callbacks to execute.

  Attributes
  ----------
  history: dict of list
    Information of training. None for now.
  """
  def __init__(self,
               model,
               train_loader,
               val_loader,
               test_loader,
               loss,
               optimizer,
              #  callbacks=None,
               *args,
               **kwargs):
    self.model = model
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.test_loader = test_loader
    self.loss = loss
    self.optimizer = optimizer
    # if not isinstance(callbacks, CallbackList):
    #   callbacks = CallbackList(callbacks)
    # self.callbacks = callbacks
    # self.callbacks.set_params({'trainer': self})
    self.history = None

  def train_step(self, *args, **kwargs):
    """
    Train the model for 1 step (or batch).
    """
    pass

  def train(self, *args, **kwargs):
    """
    Train the model.
    """
    pass

  def evaluate(self, *args, **kwargs):
    """
    Conduct prediction and evaluation.
    """
    pass

  def predict(self, *args, **kwargs):
    """
    Make prediction.
    """
    pass

class RNNTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, test_loader,
                 loss, optimizer, scaler, device, save_path, res_df):
        super(RNNTrainer, self).__init__(model = model,
                                         train_loader= train_loader,
                                         val_loader= val_loader,
                                         test_loader= test_loader,
                                         loss = loss,
                                         optimizer= optimizer
                                         )
        
        self.scaler=scaler
        self.save_path=save_path
        self.device = device
        self.res_df = res_df
    
    def train(self, epochs):

        loss_graph = []
        min_val_loss, min_val_epoch = 500, 0 # 대에충
        n = len(self.train_loader)
        for epoch in range(epochs):
            running_loss = 0.0
            
            for data in self.train_loader:
                
                # seq : [batch, time, 1], target : [batch, 1]
                seq, target = data
                out = self.model(seq)
                l = self.loss(out, target)
                
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                running_loss += l.item()
                
            loss_graph.append(running_loss/n)
            val_loss = self.evaluate()

            # Print training process
            if epoch % 100 == 0:
                print('[epoch : %d] train_loss : %.4f val_loss : %.4f'%(epoch, running_loss/n, val_loss))
                
            # Model save
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_val_epoch = epoch
                torch.save(self.model.state_dict(), self.save_path + '/save_model.pt')
            
        return min_val_loss, min_val_epoch
        
        
    def evaluate(self):
        n = len(self.val_loader)
        running_loss = 0.0
        
        with torch.no_grad():
            self.model.eval()
            for x_batch, y_batch in self.val_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(x_batch)
                l = self.loss(y_pred, y_batch)
                running_loss += l.item()
                
            self.model.train()
        return running_loss / n
    
    
    def predict(self, diff_, horizon):
        import pandas as pd
        self.model.load_state_dict(torch.load(self.save_path + '/save_model.pt'))
        with torch.no_grad():
            self.model.eval()
            val_input = self.test_loader.to(self.device)
            predictions = self.model(val_input)
            # Inverse prediction to original scale
            # predictions size : (days,) -> (days, 1)
            predictions = predictions.squeeze().clone().detach().cpu().numpy().reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions)
            # predictions size : (days,) -> (days, 1)
            predictions = predictions.reshape(-1,)
            # predictions = pd.DataFrame(predictions,
            #                             columns=self.res_df.columns,
            #                             index=self.res_df.index[-horizon:])
            
            # print(predictions.shape)
            # print(self.raw_test[:-1].shape)
            # predictions = predictions.reshape(-1,) + self.raw_test[:-1] # diff로 차이났던 부분을 여기서 바꿔줌
        
        return predictions
      
      
    def multi_step_ahead_predict(self, split_date, TIME_STEPS, horizon, region, y_test):
        # import pandas as pd
        # import numpy as np
        self.model.load_state_dict(torch.load(self.save_path + '/save_model.pt'))
        with torch.no_grad():
          self.model.eval()
          ground_truth, idx_list = [], []
          x_test = self.test_loader[0].reshape(1, TIME_STEPS, 1)
          predictions = self.test_loader[0].reshape(5, 1).tolist()
          index = self.res_df[self.res_df.index >= split_date].index[TIME_STEPS:]
          # predictions = pd.DataFrame({})#, columns=self.res_df.columns, index=[index[0]])
          # ground_truth = pd.DataFrame({})#, columns=self.res_df.columns, index=[index[0]])

          for i, index in enumerate(index[:horizon]):
              
              pred_test = self.model(x_test)
              pred_test = pred_test.squeeze().clone().detach().cpu().numpy().reshape(1,-1)#[0].tolist()
              pred_test = self.scaler.inverse_transform(pred_test)
              
              # multi step ahead x_test
              predictions.append(pred_test)
              x_test = torch.tensor(predictions[i+1:], dtype=torch.float32).reshape(1, TIME_STEPS, 1)
              
              # ground truth part
              y_true = self.scaler.inverse_transform(y_test[i].reshape(1, -1))
              
              ground_truth.append(y_true[0][0])
              idx_list.append(index)
        
        return predictions, ground_truth, idx_list