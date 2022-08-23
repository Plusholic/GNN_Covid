import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt 
import matplotlib

from datetime import timedelta
import datetime
# Collections of function

def load_data_and_set_date_state(start_date, end_date, data):
    '''
    Data load and drop NaN \n
    Setting start date and end date
    '''

    i=0
    # data = pd.read_csv('대한수학회 공유확진자DB_220321_' + str(i+1) + '.csv', low_memory=False)

    data = data[['거주시군구', '거주시도', '신고일']]#, , '연령', '발병일', '선행확진자_번호']]
    data.dropna(subset=['신고일'], axis=0, inplace=True)
    data.dropna(subset=['거주시도'], axis=0, inplace=True)
    data = data.fillna(0)
    
    idx = data.index[0]
    # print(data.loc[idx,'신고일'])
    if type(data.loc[idx,'신고일']) == str:
        if data.loc[idx,'신고일'][4] =='-':
            data['신고일'] = pd.to_datetime(data['신고일'], format="%Y-%m-%d")
        elif data.loc[idx,'신고일'][4] =='.':
            data['신고일'] = pd.to_datetime(data['신고일'], format="%Y.%m.%d")
        
    elif type(data.loc[idx,'신고일']) == np.float64:
        data['신고일'] = data['신고일'].astype(int)
        data['신고일'] = pd.to_datetime(data['신고일'], format="%Y%m%d")
        
    elif type(data.loc[idx,'신고일']) == np.int64:
        data['신고일'] = pd.to_datetime(data['신고일'], format="%Y%m%d")
    
    # print(data.loc[idx,'신고일'])
    data = data[(data['신고일'] >= pd.to_datetime(start_date, format="%Y-%m-%d")) & (data['신고일'] <= pd.to_datetime(end_date, format="%Y-%m-%d"))]
    data['확진자'] = [1]*len(data)
    return data

def load_data_and_set_date_city(start_date, end_date, data):
    '''
    Data load and drop NaN \n
    Setting start date and end date
    위에꺼랑 합치기
    '''

    i=0
    # data = pd.read_csv('대한수학회 공유확진자DB_220321_' + str(i+1) + '.csv', low_memory=False)

    data = data[['거주시군구', '거주시도', '신고일', '지역세분화']] #'발병일', '연령', '선행확진자_번호', 
    data.dropna(subset=['신고일'], axis=0, inplace=True)
    data.dropna(subset=['지역세분화'], axis=0, inplace=True)
    
    # print(data.fillna(0))
    
    data = data.fillna(0)
    
    idx = data.index[0]
    # print(data.loc[idx,'신고일'])
    if type(data.loc[idx,'신고일']) == str:
        if data.loc[idx,'신고일'][4] =='-':
            data['신고일'] = pd.to_datetime(data['신고일'], format="%Y-%m-%d")
        elif data.loc[idx,'신고일'][4] =='.':
            data['신고일'] = pd.to_datetime(data['신고일'], format="%Y.%m.%d")
        
    elif type(data.loc[idx,'신고일']) == np.float64:
        data['신고일'] = data['신고일'].astype(int)
        data['신고일'] = pd.to_datetime(data['신고일'], format="%Y%m%d")
    
    
    # print(data)
    
    data['신고일'] = pd.to_datetime(data['신고일'], format="%Y-%m-%d")
    data = data[(data['신고일'] >= pd.to_datetime(start_date, format="%Y-%m-%d")) & (data['신고일'] <= pd.to_datetime(end_date, format="%Y-%m-%d"))]
    data['확진자'] = [1]*len(data)
    return data

def make_dummy_date(start_date, end_date):
    '''
    Fill date of empty part
    '''

    start_y = int(start_date[0:4])
    start_m = int(start_date[5:7])
    start_d = int(start_date[8:10])
    end_y = int(end_date[0:4])
    end_m = int(end_date[5:7])
    end_d = int(end_date[8:10])

    diff_days = datetime.date(end_y,end_m,end_d) - datetime.date(start_y,start_m,start_d)
    
    date_list, week_list = [], []
    for i in range(diff_days.days+1): # +1 해줘야 지정한 날 마지막까지 감.
        date_list.append(datetime.date(start_y, start_m, start_d) + timedelta(days=i))
        
    for i in range(int(diff_days.days/7)):
        week_list.append(datetime.date(start_y, start_m, start_d) + timedelta(weeks=i))
        
    date_df = pd.DataFrame({'신고일' : date_list})
    week_df = pd.DataFrame({'신고주' : week_list})
    return date_df, week_df


def define_daily_dataframe(data, date_df, region, category):

    '''
    daily confirmed case and unlinked ratio
    '''
    if category == 'city':
        data = data[data['지역세분화'] == region]
    elif category == 'state':
        data = data[data['거주시도'] == region]
        
    data['신고일'] = data['신고일'].apply(lambda x: x.date())
    daily_data = pd.DataFrame({})
    daily_data['신고일'] = data['신고일'].value_counts().index
    daily_data['확진자'] = data['신고일'].value_counts().values
    
    unlinked_ratio = []
    # for c_date in daily_data['신고일']:
        
    #     # 해당 신고일에 선행확진자 번호가 0인 사람을 카운트해서 비율로 추가
    #     tmp = data['선행확진자_번호'][data['신고일'] == c_date]
    #     unlinked_ = tmp.value_counts()[0]
    #     length = len(tmp)
    #     unlinked_ratio.append(unlinked_/length)
    
    # daily_data = preprocessing_01(daily_data, date_df, unlinked_ratio, region, region_idx, '신고일')
    daily_data = preprocessing_01(daily_data, date_df, unlinked_ratio, region, '신고일')
    # daily case는 연령별 추가 안해뒀음.
    return daily_data

def define_weekly_dataframe(data, week_df, region):
    
    '''
    weekly confirmed case and unlinked ratio
    '''
    weekly_data = pd.DataFrame({})
    week_case,week_date = [], []
    unlinked_ratio = []
    age_under_20 = []
    age_under_30 = []
    age_under_40 = []
    age_under_50 = []
    age_under_60 = []
    age_under_70 = []
    age_under_80 = []
    age_etc = []
     
    data = data[data['거주시도'] == region]
    data['신고일'] = data['신고일'].apply(lambda x: x.date())
    
    for w_date in week_df['신고주']:
        week_sum = ((data['신고일'] >= w_date) & (data['신고일'] < w_date + timedelta(weeks=1))).sum()
        week_case.append(week_sum)
        week_date.append(w_date)
        
    # weekly_data['신고주'] = week_date
    # weekly_data['확진자'] = week_case
    # for w_date in weekly_data['신고주']:
        
        tmp = data[(data['신고일'] >= w_date) & (data['신고일'] < w_date + timedelta(weeks=1))]
        
        age_group_list = add_age_group(tmp)
        age_under_20.append(age_group_list[0]/week_sum)
        age_under_30.append(age_group_list[1]/week_sum)
        age_under_40.append(age_group_list[2]/week_sum)
        age_under_50.append(age_group_list[3]/week_sum)
        age_under_60.append(age_group_list[4]/week_sum)
        age_under_70.append(age_group_list[5]/week_sum)
        age_under_80.append(age_group_list[6]/week_sum)
        age_etc.append(age_group_list[7]/week_sum)
        
        unlinked_ratio = 0
        # 선행확진자 번호가 0인 사람이 없다면 unlinked ratio를 0으로, 아니라면 unlinked의 비율을 추가
        # if len(tmp[tmp['선행확진자_번호'] == 0]) == 0:
        #     unlinked_ratio.append(0)
        # else:
        #     unlinked_ = tmp['선행확진자_번호'].value_counts()[0]
        #     length = len(tmp)
        #     unlinked_ratio.append(unlinked_/length)
            
    weekly_data['신고주'] = week_date
    weekly_data['확진자'] = week_case
    weekly_data = preprocessing_01(weekly_data, week_df, unlinked_ratio, region, '신고주')
    weekly_data = preprocessing_02(weekly_data, age_under_20, age_under_30, age_under_40, age_under_50, age_under_60, age_under_70, age_under_80, age_etc)

    return weekly_data
    
def preprocessing_01(data, date_df, unlinked_ratio, region, sorting_criterion):
    ''' 
    daily, weekly data preprocessing and add unlinked case \n
    sorting_criterion = 신고일, 신고주
    '''

    # data['unlinked_ratio'] = unlinked_ratio
    # data['unlinked_ratio'].astype(float)
    
    data = pd.merge(data, date_df, how='outer')
    # data['거주시도'] = region # 번호로 해주니까 없어도 될듯
    # data['region_idx'] = region_idx # 이거도 필요없음
    data.fillna(0, inplace = True) # 날짜 합치면서 생기는 NaN 0으로 변환
    data.sort_values(by=sorting_criterion, inplace=True) # date 기준으로 sort
    data.index = [i for i in range(len(data))] # merge하면서 뒤죽박죽인 index를 다시 0부터
    # data['time_idx'] = [i for i in range(len(data))] # DeepAR 할때 필요한거
    
    # data.rename(columns = {sorting_criterion:'date', '확진자' : 'value'},inplace=True)
    data.rename(columns = {sorting_criterion:'date', '확진자' : region},inplace=True)
    del data['date']
    return data

def add_age_group(tmp):

    age_group_list = []
    age_group_list.append(len(tmp[tmp['연령'] < 20]))
    age_group_list.append(len(tmp[(tmp['연령'] < 30) & (tmp['연령'] >=20)]))
    age_group_list.append(len(tmp[(tmp['연령'] < 40) & (tmp['연령'] >=30)]))
    age_group_list.append(len(tmp[(tmp['연령'] < 50) & (tmp['연령'] >=40)]))
    age_group_list.append(len(tmp[(tmp['연령'] < 60) & (tmp['연령'] >=50)]))
    age_group_list.append(len(tmp[(tmp['연령'] < 70) & (tmp['연령'] >=60)]))
    age_group_list.append(len(tmp[(tmp['연령'] < 80) & (tmp['연령'] >=70)]))
    age_group_list.append(len(tmp[tmp['연령'] >= 80]))
    
    return age_group_list


def preprocessing_02(weekly_data, age_under_20, age_under_30, age_under_40, age_under_50, age_under_60, age_under_70, age_under_80, age_etc):
    
    weekly_data['<20'] = age_under_20
    weekly_data['<30'] = age_under_30
    weekly_data['<40'] = age_under_40
    weekly_data['<50'] = age_under_50
    weekly_data['<60'] = age_under_60
    weekly_data['<70'] = age_under_70
    weekly_data['<80'] = age_under_80
    weekly_data['>80'] = age_etc
    
    return weekly_data

def region_preprocessing(path):
    data_test = pd.read_csv(path, low_memory=False)
    data_test['지역세분화'] = (data_test['거주시도'] + ' ' + data_test['거주시군구'])

    # 예외처리
    data_test['거주시도'][data_test['거주시도'] == '대구시'] = '대구'
    data_test['거주시도'][data_test['거주시도'] == '김포시'] = '경기'
    data_test['거주시도'][data_test['거주시도'] == '부산시'] = '부산'
    # del data_test['Unnamed: 17']

    # 행정구역 인코딩
    data_list = pd.read_csv('/Users/jeonjunhwi/문서/Projects/Master_GNN/Data/행정구역리스트_2.csv')
    for col in data_list.columns:
        if col == '강원도':
            data_list[col] = data_list[col].str.rstrip('시군')
        elif col == '광역시':
            data_list[col] = data_list[col].str.rstrip('특별자치광역시')
        else:
            data_list[col] = data_list[col].str.rstrip('시구군')
    data_list = data_list.fillna(0)

    for col in data_list.columns:
        for region in data_list[col]:
            if region != 0:
                data_test['지역세분화'][data_test['지역세분화'].str.contains(region, na=False)] = region

    # 예외처리     
    data_test['지역세분화'] = data_test['거주시도'] + ' ' + data_test['지역세분화']          
    data_test['지역세분화'][data_test['지역세분화'] == '검역소 인천'] = '인천 인천'
    data_test['지역세분화'][data_test['지역세분화'] == '강원 안산'] = '경기 안산'
    
    return data_test


def sort_each_region_df(state_data, date_df, data_test, tmp_df, type):
    
    if type == 'state':
        region_dict = {'인천' : 0,
                    '서울' : 1,
                    '경기' : 2,
                    '전북' : 3,
                    '광주' : 4,
                    '전남' : 5,
                    '대구' : 6,
                    '경북' : 7,
                    '경남' : 8,
                    '충북' : 9,
                    '제주' : 10,
                    '부산' : 11,
                    '세종' : 12,
                    '강원' : 13,
                    '대전' : 14,
                    '울산' : 15,
                    '충남' : 16}
        for region in region_dict.keys():
            # region_idx = region_dict[region]
            daily_df = define_daily_dataframe(state_data, date_df, region, 'state')
            daily_df = pd.concat([tmp_df, daily_df], axis = 1)#, ignore_index=True)
            tmp_df = daily_df
        
    elif type == 'city':
        region_dict = dict(data_test['지역세분화'].value_counts())
            
        for region in region_dict.keys():
            # region_idx = region_dict[region]
            daily_df = define_daily_dataframe(state_data, date_df, region, 'city')
            daily_df = pd.concat([tmp_df, daily_df], axis = 1)#, ignore_index=True)
            tmp_df = daily_df

    # 결측치 채우기, 비어있는 날짜 데이터와 합치기
    daily_df = daily_df.fillna(0.0)
    daily_df = pd.concat([date_df, daily_df], axis=1)
    
    return daily_df

def merge_each_df(df, date_df, path, type):
    import os
    from glob import glob
    '''
    df = daily_df_state or daily_df_city
    '''
    print(os.getcwd())
    # folders = os.listdir(path)
    
    folders = glob(f"{path}/**{type}.csv", recursive=False)
    
    # city, state 따로 폴더 만들기
    # 점화식 초기값을 전체 날짜로 주고
    # print(folders)
    data = pd.DataFrame(0, index=[i for i in range(len(df))], columns=df.columns)
    del data['신고일']
    for files in folders:
        print(files)
        # if files[-5:-4] == 'e':
        # # 로드해서 데이터프레임을 싹다 더하기
        #     # tmp = pd.read_csv(files, encoding='euc-kr', index_col = '신고일').astype(int)
        #     tmp = pd.read_csv(f"{path}/{files}", encoding='euc-kr', index_col = 0)
        #     # print(files, tmp)
        #     # del tmp['Unnamed: 0']
        #     del tmp['신고일']
        #     tmp = tmp.astype(int)
            
        #     data = data + tmp
            
        # elif files[-5:-4] == 'y':
        # # 로드해서 데이터프레임을 싹다 더하기
        #     # tmp = pd.read_csv(files, encoding='euc-kr', index_col = '신고일').astype(int)
        #     tmp = pd.read_csv(f"{path}/{files}", encoding='euc-kr', index_col = 0)
        #     # print(files, tmp)
        #     # del tmp['Unnamed: 0']
        #     del tmp['신고일']
        #     tmp = tmp.astype(int)
            
        #     data = data + tmp
        # 로드해서 데이터프레임을 싹다 더하기
        # tmp = pd.read_csv(files, encoding='euc-kr', index_col = '신고일').astype(int)
        tmp = pd.read_csv(f"{files}", encoding='euc-kr', index_col = 0)
        del tmp['신고일']
        tmp = tmp.astype(int)
        
        data = data + tmp
    data['신고일'] = date_df
    data = data.set_index('신고일')
    data.to_csv(f"./Processing_Results/conf_data_{type}.csv", encoding='cp949')

    return data

def smoothing_and_save(df, smoothing_window, path, type, df_save, fig_save):
    
    # plot에서 한글 폰트 깨지는 현상 해결!
    from matplotlib import font_manager, rc
    'Apple SD Gothic Neo'
    font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family = font)
    
    # smoothing window만큼 rolling
    df = df.rolling(smoothing_window).sum().dropna()
    # save smoothing dataframe
    if df_save == True:
        df.to_csv(path + f"/smoothing_{smoothing_window}_{type}.csv", encoding='cp949')
    
    if fig_save == True:
        fig = plt.figure(figsize=(20,15))
        for i, region in enumerate(list(df.columns)):
            ax = fig.add_subplot(5,4,i+1)
            df[region].plot(x = '신고일', y = region, ax = ax)
            plt.title(region)
        
        plt.tight_layout()
        plt.savefig(path + f"/smoothing_{smoothing_window}_{type}.png")

    
    
    return df