
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt 
import matplotlib

from datetime import timedelta
import datetime
# Collections of function

def load_data_and_set_date_state(data, start_date, end_date):
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
    return data, list(data['거주시도'].unique())

def load_data_and_set_date_city(data, start_date, end_date):
    '''
    Data load and drop NaN \n
    Setting start date and end date
    위에꺼랑 합치기
    '''

    i=0
    # data = pd.read_csv('대한수학회 공유확진자DB_220321_' + str(i+1) + '.csv', low_memory=False)

    # 필요한 컬럽 추출, 필요한 부분에서 없는 부분은 0으로
    data = data[['거주시군구', '거주시도', '신고일', '지역세분화']] #'발병일', '연령', '선행확진자_번호', 
    data.dropna(subset=['신고일'], axis=0, inplace=True)
    data.dropna(subset=['지역세분화'], axis=0, inplace=True)
    data = data.fillna(0)


    # 데이터프레임의 첫 번째 인덱스를 추출해서 날짜 형식을 변경(데이터의 날짜 표현 형식이 다양함)
    idx = data.index[0]
    if type(data.loc[idx,'신고일']) == str:
        if data.loc[idx,'신고일'][4] =='-':
            data['신고일'] = pd.to_datetime(data['신고일'], format="%Y-%m-%d")
        elif data.loc[idx,'신고일'][4] =='.':
            data['신고일'] = pd.to_datetime(data['신고일'], format="%Y.%m.%d")
        
    elif (type(data.loc[idx,'신고일']) == np.float64) | (type(data.loc[idx,'신고일']) == np.int64):
        data['신고일'] = data['신고일'].astype(int)
        data['신고일'] = pd.to_datetime(data['신고일'], format="%Y%m%d")
    
    # 22-03-08 의 형식으로 날자 통일, 데이터에서 시작 날짜와 마지막 날짜 사이의 데이터만 추출
    data['신고일'] = pd.to_datetime(data['신고일'], format="%Y-%m-%d")
    data = data[(data['신고일'] >= pd.to_datetime(start_date, format="%Y-%m-%d")) & (data['신고일'] <= pd.to_datetime(end_date, format="%Y-%m-%d"))]
    data['확진자'] = [1]*len(data)
    

    # 지역별로 연속적으로 나오도록 정렬
    reg = ['인천', '서울', '경기', '전북', '광주', '전남', '대구', '경북', '경남', '부산', '울산', '충북', '대전', '충남','세종', '제주', '강원']
    new_city_list = []
    for i in reg:
        for j in list(data['지역세분화'].unique()):
            if j[:2] == i:
                new_city_list.append(j)
    
    
    return data, new_city_list

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
    # week_df = pd.DataFrame({'신고주' : week_list})
    return date_df


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
    
def preprocessing_01(data, date_df, unlinked_ratio, region, sorting_criterion):
    ''' 
    daily, weekly data preprocessing and add unlinked case \n
    sorting_criterion = 신고일, 신고주
    '''

    # data['unlinked_ratio'] = unlinked_ratio
    # data['unlinked_ratio'].astype(float)
    # print(data)
    # print(date_df)
    # input("press enter : ")
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


def region_preprocessing(path):
    data_test = pd.read_csv(path, low_memory=False)

    # 예외처리
    data_test['거주시도'][data_test['거주시도'] == '대구시'] = '대구'
    data_test['거주시도'][data_test['거주시도'] == '김포시'] = '경기'
    data_test['거주시도'][data_test['거주시도'] == '부산시'] = '부산'
    # del data_test['Unnamed: 17']

    # 행정구역 인코딩
    # data_list = pd.read_csv('/Users/jeonjunhwi/문서/Projects/Master_GNN/Data/행정구역리스트_수도권.csv')
    data_list = pd.read_csv('/Users/jeonjunhwi/문서/Projects/Master_GNN/Data/행정구역리스트_2.csv')
    # print(data_list.columns)
    
    data_test = data_test[data_test['거주시도'].isin(data_list.columns)]
    # input("press enter : ")
    # columns에서 오른쪽 끝의 ~시, ~구 제거
    # 수원시 -> 수원, 인천광역시 -> 인천
    # for col in data_list.columns:
    #     if col == '강원도':
    #         data_list[col] = data_list[col].str.rstrip('시군')
    #     elif col == '광역시':
    #         data_list[col] = data_list[col].str.rstrip('특별자치광역시')
    #     else:
    #         data_list[col] = data_list[col].str.rstrip('시구군')
            
    # for col in data_list.columns:
    #     if col == '서울특별시':
    #         data_list[col] = data_list[col].str.rstrip('시군')
    #     elif col == '광역시':
    #         data_list[col] = data_list[col].str.rstrip('특별자치광역')
    #     else:
    #         data_list[col] = data_list[col].str.rstrip('시구군')
            

    data_test['지역세분화'] = (data_test['거주시도'] + ' ' + data_test['거주시군구'])
    # data_list = data_list.fillna(0)
    ##############
    ### 수도권 ###
    ##############
    data_test['지역세분화'][data_test['지역세분화'].str.contains('일산', na=False)] = '경기 고양시'
    data_test['지역세분화'][data_test['지역세분화'].str.contains('김포', na=False)] = '경기 김포시'
    data_test['지역세분화'][data_test['지역세분화'].str.contains('수원', na=False)] = '경기 수원시'
    data_test['지역세분화'][data_test['지역세분화'].str.contains('용인', na=False)] = '경기 용인시'
    data_test['지역세분화'][data_test['지역세분화'].str.contains('걸포', na=False)] = '경기 김포시'
    data_test['지역세분화'][data_test['지역세분화'].str.contains('부천', na=False)] = '경기 부천시'
    data_test['지역세분화'][data_test['지역세분화'].str.contains('용산', na=False)] = '서울 용산구'
    data_test['지역세분화'][data_test['지역세분화'].str.contains('안양', na=False)] = '경기 안양시'
    data_test['지역세분화'][data_test['지역세분화'].str.contains('성남', na=False)] = '경기 성남시'
    data_test['지역세분화'][data_test['지역세분화'].str.contains('안산', na=False)] = '경기 안산시'
    data_test['지역세분화'][data_test['지역세분화'].str.contains('고양', na=False)] = '경기 고양시'
    data_test['지역세분화'][data_test['지역세분화'].str.contains('인천 남구', na=False)] = '인천 미추홀구'
    
    ##############
    ### 충청도 ###
    ##############
    data_test['지역세분화'][data_test['지역세분화'].str.contains('세종', na=False)] = '세종 세종시'
    data_test['지역세분화'][data_test['지역세분화'].str.contains('청주', na=False)] = '충북 청주시'
    data_test['지역세분화'][data_test['지역세분화'].str.contains('천안', na=False)] = '충남 천안시'
    
    ##############
    ### 경상도 ###
    ##############
    data_test['지역세분화'][data_test['지역세분화'].str.contains('창원', na=False)] = '경남 창원시'
    data_test['지역세분화'][data_test['지역세분화'].str.contains('포항', na=False)] = '경북 포항시'
    
    ##############
    ### 전라도 ###
    ##############
    data_test['지역세분화'][data_test['지역세분화'].str.contains('전주', na=False)] = '전북 전주시'
    
    # print(data_list)
    # input("press enter : ")
    
    # for col in data_list.columns:
    #     # print(col)
    #     if col == '서울':
    #         data_list[col] = data_list[col].str.rstrip('시군')
    #     else:
    #         data_list[col] = data_list[col].str.rstrip('시구군')
    #     for region in data_list[col].fillna(0):
    #         if region != 0:
    #             data_test['지역세분화'][data_test['지역세분화'].str.contains(region, na=False)] = str(col) + ' ' + str(region)
    # input("press enter : ")
    
    
    return data_test


def sort_each_region_df(state_data, region_list, date_df, data_test, tmp_df, category):
    tmp_df = pd.DataFrame({})

    for region in region_list:
        # region_idx = region_dict[region]
        daily_df = define_daily_dataframe(state_data, date_df, region, category)
        daily_df = pd.concat([tmp_df, daily_df], axis = 1)#, ignore_index=True)
        tmp_df = daily_df

    # 결측치 채우기, 비어있는 날짜 데이터와 합치기
    daily_df = daily_df.fillna(0.0)
    daily_df = pd.concat([date_df, daily_df], axis=1)
    
    return daily_df

def merge_each_df(date_df, region_list, path, category):
    import os
    from glob import glob
    '''
    df = daily_df_state or daily_df_city
    '''
    print(os.getcwd())
    # folders = os.listdir(path)
    
    folders = glob(f"{path}/**{category}.csv", recursive=False)
    
    # city, state 따로 폴더 만들기
    # 점화식 초기값을 전체 날짜로 주고
    # print(folders)
    data = pd.DataFrame(0, index=[i for i in range(len(date_df))], columns=region_list)
    # del data['신고일']
    for files in folders:

        tmp = pd.read_csv(f"{files}", encoding='euc-kr', index_col = 0)
        del tmp['신고일']
        print(files)
        print(len(region_list))
        print(tmp.shape)
        print(list(region_list))
        # print(files)
        # print(tmp)
        tmp = tmp[list(region_list)]
        tmp = tmp.astype(int)
        # print(tmp.head())
        data = data + tmp
        # print(data.head())
    input("press enter : ")    
    
    data['신고일'] = date_df
    data = data.set_index('신고일')
    data.to_csv(f"./Processing_Results/conf_case_{category}.csv", encoding='cp949')

    return data

def smoothing_and_save(df, smoothing_window, path, category, df_save, fig_save):
    
    # plot에서 한글 폰트 깨지는 현상 해결!
    from matplotlib import font_manager, rc
    'Apple SD Gothic Neo'
    font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family = font)
    
    # smoothing window만큼 rolling
    df = df.rolling(smoothing_window).mean().dropna()
    # save smoothing dataframe
    if df_save == True:
        df.to_csv(path + f"/smoothing_{smoothing_window}_{category}.csv", encoding='cp949')
    
    if fig_save == True:
        fig = plt.figure(figsize=(20,15))
        for i, region in enumerate(list(df.columns)):
            ax = fig.add_subplot(5,4,i+1)
            df[region].plot(x = '신고일', y = region, ax = ax)
            plt.title(region)
        
        plt.tight_layout()
        plt.savefig(path + f"/smoothing_{smoothing_window}_{category}.png")

    
    
    return df