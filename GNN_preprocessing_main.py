import warnings
warnings.filterwarnings('ignore')
import os
from GNN_preprocessing_func import *
from tqdm import tqdm

# 작업이 KCDC_data folder 내에서만 일어나도록 고정
os.getcwd()
os.chdir('./Data/KCDC_data/')
folders = os.listdir()


for files in tqdm(folders, desc = 'transform and merging'):

    if files[-3:] == 'csv':
        data_test = region_preprocessing(path=files)
        state_data = load_data_and_set_date_state('2020-01-19','2022-08-15', data_test)
        city_data = load_data_and_set_date_city('2020-01-19','2022-08-15', data_test)
        date_df, week_df = make_dummy_date('2020-01-19','2022-08-15')
        tmp_df_state, tmp_df_city = pd.DataFrame({}), pd.DataFrame({})


        
        #####################
        ## STATE 별로 저장 ##
        #####################
        
        # region_dict = {'인천' : 0,
        #             '서울' : 1,
        #             '경기' : 2,
        #             '전북' : 3,
        #             '광주' : 4,
        #             '전남' : 5,
        #             '대구' : 6,
        #             '경북' : 7,
        #             '경남' : 8,
        #             '충북' : 9,
        #             '제주' : 10,
        #             '부산' : 11,
        #             '세종' : 12,
        #             '강원' : 13,
        #             '대전' : 14,
        #             '울산' : 15,
        #             '충남' : 16}
        
        # for region in region_dict.keys():
        #     # region_idx = region_dict[region]
        #     daily_df_state = define_daily_dataframe(state_data, date_df, region, 'state')
        #     daily_df_state = pd.concat([tmp_df_state, daily_df_state], axis = 1)#, ignore_index=True)
        #     tmp_df_state = daily_df_state

        # # 결측치 채우기, 비어있는 날짜 데이터와 합치기
        # daily_df_state = daily_df_state.fillna(0.0)
        # daily_df_state = pd.concat([date_df, daily_df_state], axis=1)

        # ######################
        # ### CITY 별로 저장 ###
        # ######################
        
        # region_dict = dict(data_test['지역세분화'].value_counts())
        # for region in region_dict.keys():
        #     # region_idx = region_dict[region]
        #     daily_df_city = define_daily_dataframe(city_data, date_df, region, 'city')
        #     daily_df_city = pd.concat([tmp_df_city, daily_df_city], axis = 1)#, ignore_index=True)
        #     tmp_df_city = daily_df_city
        
        # # 결측치 채우기, 비어있는 날짜 데이터와 합치기
        # daily_df_city = daily_df_city.fillna(0.0)
        # daily_df_city = pd.concat([date_df, daily_df_city], axis=1)
        
        daily_df_state = sort_each_region_df(state_data, date_df, data_test, tmp_df = pd.DataFrame({}), type='state')
        daily_df_city = sort_each_region_df(city_data, date_df, data_test, tmp_df = pd.DataFrame({}), type='city')
        # print(daily_df_state.columns)
        daily_df_state.to_csv(f'Processing_data/{files[:-4]}_state.csv', encoding="euc-kr")
        daily_df_city.to_csv(f'Processing_data/{files[:-4]}_city.csv', encoding="euc-kr")
        
final_data = merge_each_df(daily_df_state, date_df, './Processing_data')

final_data.to_csv('state_conf_data.csv', encoding='cp949')