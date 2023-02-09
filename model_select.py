import pickle


def model_selection(MODEL_NAME = None,
                    adj_mx = None,
                    TIME_STEPS = None,
                    device=None,
                    save_path = None,
                    num_hop=1):  # ,
                    # dropedge_savename=None,
                    # dropedge_networkpath=None):

    ###### ASTGCN MODEL ######
    if MODEL_NAME == 'ASTGCN':
        from models import make_model
        print(MODEL_NAME)
        # nb_block = 1 # block 갯수 : 이거로 hop을 생각할 수 있을듯. layer의 개념이니까.
        # K = 3 # 확인해봐야함

        config = dict({'num_of_vertices' : adj_mx.shape[0],
                    #  'points_per_hour' : 12,
                     'num_for_predict' : 1,
                     'in_channels' : 1,
                     'nb_block' : num_hop,
                     'K' : 3,
                     'nb_chev_filter' : 16,
                     'nb_time_filter' : 16,
                     'time_strides' : 1})
        
        model = make_model(DEVICE=device,
                           nb_block=config['nb_block'],
                           in_channels=config['in_channels'],
                           K = config['K'],
                           nb_chev_filter=config['nb_chev_filter'],
                           nb_time_filter=config['nb_time_filter'],
                           time_strides=config['time_strides'],
                           adj_mx=adj_mx.numpy(),
                           num_for_predict=config['num_for_predict'],
                           len_input=TIME_STEPS,
                           num_of_vertices=config['num_of_vertices'])
    
    ###### TGCN MODEL ######    
    if MODEL_NAME == 'TGCN':
        print(MODEL_NAME)
        from models import TGCNConv
        
        config = dict({'hidden_dim' : 16,
                     'out_dim' : 16,
                     'num_hop' : 2})
        
        model = TGCNConv(adj_mx = adj_mx,
                        hidden_dim=config['hidden_dim'],
                        out_dim=config['out_dim'],
                        num_hop=config['num_hop'])
    
    ###### STGNN MODEL ######
    if MODEL_NAME == 'STGNN':
        print(MODEL_NAME)
        from models import ProposedSTGNN
        
        config = dict({'predicted_time_steps' : 1,
                     'in_channels' : 1,
                     'spatial_channels' : [16],# 16],
                     'temporal_kernel' : 3,
                     'FourierEmbedding' : False,
                     'drop_rate' : 0.2})
        
        model = ProposedSTGNN(n_nodes=adj_mx.shape[0],
                              adj_mx=adj_mx,
                              time_steps=TIME_STEPS,
                              predicted_time_steps=config['predicted_time_steps'],
                              in_channels=config['in_channels'],
                              spatial_channels=config['spatial_channels'],
                              temporal_kernel=config['temporal_kernel'],
                            #   FourierEmbedding=True,
                              FourierEmbedding=False,
                              gnn_norm=False,
                              drop_rate=config['drop_rate']).to(device=device)

    ###### GCN MODEL ######
    if MODEL_NAME == 'GCN':
        from models import GCN
        print(MODEL_NAME)

        config = dict({'hidden_feats': [64 for _ in range(num_hop)],  # , 64],
                       'activation': None,
                       'residual': None,
                       'batchnorm': None,
                       'dropout': [0.2 for _ in range(num_hop)],
                       'gnn_norm': None})
        
        model = GCN(in_feats=1,
                    hidden_feats=config['hidden_feats'],
                    adj_mx = adj_mx,
                    TIME_STEPS=TIME_STEPS,
                    activation=config['activation'],
                    residual=config['residual'],
                    batchnorm=config['batchnorm'],
                    dropout=config['dropout'],
                    gnn_norm=config['gnn_norm'],
                    FourierEmbedding=True,
                    # FourierEmbedding=False,
                    device=device).to(device=device)
        
    ###### DCRNN MODEL ######
    if MODEL_NAME == 'DCRNN':
        from models import DCRNNModel
        config = dict({'rnn_units' : 64,
                     'num_rnn_layers' : num_hop,
                     'horizon' : 1,
                     'max_diffusion_step' : 5,
                     'cl_decay_steps' : 1000,
                     'filter_type' : "dual_random_walk",#'laplacian',#'dual_random_walk',
                     'use_curriculum_learning' : False})
        
        model = DCRNNModel(adj_mx=adj_mx,
                        input_dim=1,
                        output_dim=1, # 확인필요
                        rnn_units=config['rnn_units'],
                        num_rnn_layers=config['num_rnn_layers'],
                        seq_len=TIME_STEPS,
                        horizon=config['horizon'],
                        max_diffusion_step=config['max_diffusion_step'],
                        cl_decay_steps=config['cl_decay_steps'],
                        filter_type=config['filter_type'],
                        num_nodes=adj_mx.shape[0],
                        use_curriculum_learning=config['use_curriculum_learning']).to(device=device)

    ###### GCN2 MODEL ######
    if MODEL_NAME == 'GCN2':
        print(MODEL_NAME)
        from models import GCNII
        config = dict({'nlayers' : 1,
                     'nhidden' : 16,
                     'dropout' : 0.2,
                     'lambda' : 0.5,
                     'alpha' : 0.1,
                     'nclass' : 1,
                     'variant':False})
        model = GCNII(adj_mx = adj_mx,
                    TIME_STEPS = TIME_STEPS,
                    nfeat=1,
                    nlayers=config['nlayers'],
                    nhidden=config['nhidden'],
                    nclass=config['nclass'], #확인필요
                    dropout=config['dropout'],
                    lamda=config['lambda'],
                    alpha=config['alpha'],
                    variant=config['variant'],
                    FourierEmbedding=True).to(device=device)

    ###### STGCN MODEL ######
    if MODEL_NAME == 'STGCN':
        print(MODEL_NAME)
        from models import STGCN_WAVE
        import networkx as nx
        import dgl
        
        config = dict({'control_str' : 'TSNT', # T로 끝나면 ([16, 1, 1, 229])
                       'blocks' : [1, 16, 16, 16, 16, 16],
                       'drop_prob' : 0})
        
        G = dgl.from_networkx(nx.Graph(adj_mx.numpy()))
        model = STGCN_WAVE(c=config['blocks'],
                           T=TIME_STEPS,
                           n=adj_mx.shape[0], #num_node
                           Lk=G,
                           p=config['drop_prob'], # 이거 작동안함
                           num_layers=len(config['control_str']),
                           device=device,
                           control_str=config['control_str']).to(device)

    if MODEL_NAME == 'LSTM':
        from models import LSTM

        config = {'hidden_size' : 16,
                  'num_layers' : 1}

        model = LSTM(input_size = 1,
                    hidden_size = config['hidden_size'],
                    sequence_length = TIME_STEPS,
                    num_layers = config['num_layers'],
                    device = device).to(device)


    with open(f'{save_path}/config.pkl', 'wb') as f:
        pickle.dump(config, f)
    return model


