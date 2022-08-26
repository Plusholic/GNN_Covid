import torch
import numpy as np

def matplotlib_plot_font():

    # plot에서 한글 폰트 깨지는 현상 해결!
    from matplotlib import font_manager, rc
    font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family = font)


def evaluate_model(model, loss, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    cum_pred = torch.tensor([])
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
            
            cum_pred = torch.cat((cum_pred, y_pred),0)
            # print(y_pred.shape)
        return l_sum / n, cum_pred


def evaluate_metric(model, data_iter, scaler):
    model.eval()
    with torch.no_grad():
        mae, mape, mse = [[] for i in range(17)], [[] for i in range(17)], [[] for i in range(17)]
        cum_pred = torch.tensor([])
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)
            d = np.abs(y - y_pred)
            
            for i in range(len(d)):
                mae[i].append(d[i])
                mape[i].append(d[i] / y[i])
                mse[i].append(d[i] ** 2)
                 
            # mae += d.tolist()
            # mape += (d / y).tolist()
            # mse += (d ** 2).tolist()
            
            y_pred = torch.tensor(y_pred)
            cum_pred = torch.cat((cum_pred, y_pred),0)
        

        # 위에서 지역별로 추가해준 각각의 평균을 구해줌(axis=1)
        MAE = np.array(mae).mean(axis=1)
        MAPE = np.array(mape).mean(axis=1)
        RMSE = np.sqrt(np.array(mse).mean(axis=1))
        
        return MAE, MAPE, RMSE, cum_pred

def test_rnn(model, test_loader):
    with torch.no_grad():
        pred = []
        model.eval()
        for data in test_loader:
            seq, target = data
            out = model(seq)
            pred += out.cpu().tolist()
    return pred

    
def save_figure_predict(df, y_pred,
                        gru_res=None, lstm_res=None,
                        len_train=None, len_val=None, n_his_pred=None,
                        region_dict=None, suptitle = None,
                        MAE=None, MAPE=None, RMSE=None,
                        PATH=None):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(25,15), facecolor='white')
    pred_val = []
    
    for i in range(y_pred.shape[1]): # 17개
        pred_val.append([])
    for i in range(y_pred.shape[0]): # test 날짜
        for j in range(y_pred.shape[1]):
            pred_val[j].append(y_pred[i][j])
    
    for i in range(y_pred.shape[1]): # 17 도시에 대해서 각 도
        fig.add_subplot(4,5,i+1)
        
        
        pred_val[i] = [None]*n_his_pred + pred_val[i]
        plt.plot(list(df.iloc[len_train + len_val:,i].values), '--')
        plt.plot(pred_val[i], 'g', linewidth=0.6) # GNN result

        if gru_res is not None:
            plt.plot(gru_res[str(i)], 'b', linewidth=0.6) # GRU result
        if lstm_res is not None:
            plt.plot(lstm_res[str(i)], 'r', linewidth=0.6) # LSTM result
        
        title_ = f"{region_dict[i]} \n MAE: {MAE[i]:.4f}, MAPE: {MAPE[i]:.4f}, RMSE : {RMSE[i]:.4f}"
        plt.title(title_, fontsize=15)
        plt.legend(['ground truth', 'STGCN', 'GRU', 'LSTM'])
        # xlabels = list(df.index[len_train + len_val:])
        xlabels = [i[5:] for i in list(df.index[len_train + len_val:])]
        plt.xticks(ticks = [i for i in range(len(xlabels))], labels = xlabels, rotation=90)
        plt.yticks(fontsize = 15)

    plt.suptitle(suptitle, fontsize=30)
    plt.tight_layout()
    plt.savefig(f"{PATH}_{suptitle}.png")
    import pandas as pd
    pd.DataFrame({'MAE' : MAE,
                  'MAPE' : MAPE,
                  'RMSE' : RMSE,
                  }, index=region_dict.values()).to_csv(f'{PATH}_{suptitle}_summary.csv', encoding='cp949')