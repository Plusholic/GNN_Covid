from keras.callbacks import CallbackList
import torch
import numpy as np
import math
import time

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
               scheduler=None,
               callbacks=None,
               *args,
               **kwargs):
    self.model = model
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.test_loader = test_loader
    self.loss = loss
    self.optimizer = optimizer
    self.scheduler = scheduler
    if not isinstance(callbacks, CallbackList):
      callbacks = CallbackList(callbacks)
    self.callbacks = callbacks
    self.callbacks.set_params({'trainer': self})
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

class Trainer(BaseTrainer):
    
  def __init__(self,
               model,
               train_loader,
               val_loader,
               test_loader,
               scaler,
               loss,
               optimizer,
               save_path,
               device,
               res_df=None,
               scheduler=None,
               *args,
               **kwargs):

    super(Trainer, self).__init__(model,
                                train_loader,
                                val_loader,
                                test_loader,
                                loss,
                                optimizer,
                                scheduler,
                                callbacks=None)
    if res_df is None:
      raise Exception('No residual data.')
    self.res_df = res_df
    self.scaler = scaler
    self.device = device
    self.save_path = save_path

  def train(self, epochs):#, dropedge, dropedge_dict):
    """
    Train the model.
    """


    self.history = {'epoch': [],
                    'train_loss': [],
                    'val_loss': [],
                    'elapsed_time': [],
                    'learning_rate' : []}
    min_val_loss = np.inf
    start_train_time = time.time()
    
    
    for epoch in range(epochs):
      # lrs= []
      train_loss = 0.0
      
      # 1. 여기에 DropEdge를 넣는 방안
      for x_batch, y_batch in self.train_loader:
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        # 2. 여기에 DropEdge를 넣는 방안
        loss = self.train_step(x_batch, y_batch)
        train_loss += loss.item()

      if self.scheduler is not None:
        self.scheduler.step()
        
      # calculate validation loss
      val_loss = self.evaluate()
      if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(self.model.state_dict(), self.save_path + '/save_model.pt')
        self.save_epoch = epoch
        print(min_val_loss, self.save_path)
        
        
      end_train_time = time.time()
      self.history['epoch'].append(epoch + 1)
      self.history['train_loss'].append(train_loss / len(self.train_loader))
      self.history['val_loss'].append(val_loss)
      self.history['elapsed_time'].append(end_train_time - start_train_time)
      self.history['learning_rate'].append(self.optimizer.param_groups[0]["lr"])
      msg = 'Epoch: {} : Elapsed time: {:.4f}, Train loss: {:.4f}, Val loss: {:.4f}'
      print(msg.format(self.history['epoch'][-1],
                       self.history['elapsed_time'][-1],
                       self.history['train_loss'][-1],
                       self.history['val_loss'][-1]))
                      #  math.sqrt(self.history['val_loss'][-1])))
    
    # loss plot 저장
    self.loss_plot()
    # history dictionary 저장
    import pickle
    with open(f'{self.save_path}/train_history.pkl', 'wb') as f:
        pickle.dump(self.history, f)

  def train_step(self, x_batch, y_batch):

    self.model.train()
    self.optimizer.zero_grad()
    y_pred = self.model(x_batch)
    
    loss = self.loss(y_pred, y_batch)
    loss.backward()
    self.optimizer.step()
    return loss#.detach()


  def evaluate(self, *args, **kwargs):
      
    self.model.eval()
    l_sum, n = 0.0, 0
    # cum_pred = torch.tensor([]).to(self.device)
    with torch.no_grad():
        for x_batch, y_batch in self.val_loader:
            
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            y_pred = self.model(x_batch)
            l = self.loss(y_pred, y_batch)
            l_sum += l.item()
            
        self.model.train()
        return l_sum / len(self.val_loader)

  def predict(self, diff_, horizon, *args, **kwargs):
    import pandas as pd
    self.model.load_state_dict(torch.load(self.save_path + '/save_model.pt'))
    with torch.no_grad():
      self.model.eval()
      val_input = self.test_loader[0].to(self.device)
      predictions = self.model(val_input)
      predictions = predictions.squeeze().detach().cpu().numpy()
      predictions = self.scaler.inverse_transform(predictions)
      
      # prediction은 t-1일과 t일의 차이를 t일에 저장함.
      # t일의 prediction + t-1일의 데이터를 더해줘야 t일이 예측됨.
      # diff는 맨 뒤의 데이터의 날짜는 변화하지 않고, 맨 앞의 날짜만 NaN으로 변함.
      # 원래 데이터(res_df)는 하루씩 땡겨서 더해줘야함.
      
      # if diff_ == "1st":
      predictions = pd.DataFrame(predictions,
                                  columns=self.res_df.columns,
                                  index=self.res_df.index[-horizon:])
      return predictions
      
      
  def multi_step_ahead_predict(self, split_date, TIME_STEPS, horizon):
    import pandas as pd
    self.model.load_state_dict(torch.load(self.save_path + '/save_model.pt'))
    with torch.no_grad():
      self.model.eval()
          
      x_test = self.test_loader[0][0].reshape(1, 229, 5, 1)

      index = self.res_df[self.res_df.index >= split_date].index[TIME_STEPS:]
      predictions = pd.DataFrame({}, columns=self.res_df.columns, index=[index[0]])
      ground_truth = pd.DataFrame({}, columns=self.res_df.columns, index=[index[0]])

      for i, index in enumerate(index[:horizon]):
          
          # (node, timestep, 1) -> (1, node, timestep, 1)
          pred_test = self.model(x_test) 
          # (1, node, 1) -> (1, node)
          pred_test = pred_test.squeeze().detach().cpu().numpy().reshape(1,-1) 
          
          x_test = pd.DataFrame(x_test.transpose(1, 0).reshape(-1, 229).numpy(), columns=self.res_df.columns)
          x_test = x_test.drop(0) # 첫번째 열 드랍
          x_test.loc[-1,:] = pred_test
          # (timestep, node) -> (1, node, timestep, 1)
          x_test = x_test.to_numpy().transpose(1, 0).reshape(1, 229, 5, 1)
          x_test = torch.tensor(x_test, dtype=torch.float32)
          
          # (1, node) -> (1, node)
          pred_test = self.scaler.inverse_transform(pred_test)
          y_true = self.scaler.inverse_transform(self.test_loader[1][i].reshape(1, -1))
          y_true = np.array(y_true)#, dtype='float32')
          predictions.loc[index,:] = pred_test
          ground_truth.loc[index,:] = y_true

      ground_truth = ground_truth.astype(float) 
      predictions = predictions.astype(float)
        
    return predictions, ground_truth
  
  
  def loss_plot(self):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,10))
    fig.add_subplot(2,1,1)
    plt.plot(self.history['train_loss'])
    plt.plot(self.history['val_loss'])
    plt.legend(['train', 'validation'])
    plt.title("Loss Plot, " + str(self.history['elapsed_time'][-1])[:5] + "sec")
    plt.xlabel('epoch')
    plt.vlines(self.save_epoch, 0, max(max(self.history['train_loss']), max(self.history['val_loss'])))
    
    fig.add_subplot(2,1,2)
    plt.plot(self.history['learning_rate'])
    plt.title('Learning Rate')
    plt.savefig(self.save_path + '/train_plot.png')
    

# class DropEdge:
#     def __init__(self, adj_mx, percent):
#         self.adj_mx = adj_mx
#         self.percent = percent
        
#     def _dropedge(self):
#         import numpy as np
#         print('number of edges before DropEdge: ', self.adj_mx[self.adj_mx>0].__len__())

#         nnz =self.adj_mx.nonzero() # 연결되어있는 엣지 쌍을 반환
#         perm = np.random.permutation(nnz) # 엣지 쌍의 순서를 섞어줌
#         drop_nnz = int(len(nnz)*self.percent) # 엣지 쌍의 길이에 드롭 할 퍼센트를 곱해줌
#         perm = perm[:drop_nnz] # 드롭할 만큼의 엣지 쌍을 반환

#         row = perm.transpose(1,0)[0] # 전치 후 첫번째 열은 source node
#         col = perm.transpose(1,0)[1] # 전치 후 두번째 열은 target node 
#         self.adj_mx[row, col] = 0 # 이 부분을 0으로 drop 해줌

#         print('number of edges after DropEdge : ', self.adj_mx[self.adj_mx>0].__len__())
        
#         # tensor의 형태로 adj_mx를 반환
#         return self.adj_mx

#     def _save(self, Network_path, save_name, cnt):
#         import pandas as pd
#         import matplotlib.pyplot as plt
#         import seaborn as sns
#         tmp = pd.DataFrame(self.adj_mx.numpy())#, columns = dist_mx.columns, index = dist_mx.index)
#         plt.figure(figsize=(30,30))
#         g=sns.heatmap(tmp) #annot=True,

#         cbar = g.collections[0].colorbar
#         cbar.ax.tick_params(labelsize=30)

#         plt.yticks(fontsize=15, rotation=0)
#         plt.xticks(fontsize=15, rotation=90)
#         plt.savefig(f"{Network_path}/{save_name}_drop_{self.percent}_{cnt}.png")