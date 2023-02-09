from operator import index
from unicodedata import category
import warnings
warnings.filterwarnings('ignore')
import os
from data_preprocessing_func import *
from tqdm import tqdm
from glob import glob

# 작업이 KCDC_data folder 내에서만 일어나도록 고정
os.getcwd()
os.chdir('./Data/KCDC_data/')
# folders = os.listdir()

# 폴더에서 csv 확장자만 가져오기
folders = glob("**.csv", recursive=False)
start_year, start_month, start_date = 2020, 1, 19
end_year, end_month, end_date = 2022, 8, 15

start_date = f"{start_year}-{str(start_month).zfill(2)}-{str(start_date).zfill(2)}"
end_date = f"{end_year}-{str(end_month).zfill(2)}-{str(end_date).zfill(2)}"


# print(folders)
for i, files in enumerate(tqdm(folders, desc = 'transform and merging')):
# for i, files in enumerate(tqdm(['대한수학회 공유확진자DB_200119-211031 기준_제공.csv'])):
# for i, files in enumerate(tqdm(['대한수학회 공유확진자DB_200119-211031 기준_제공.csv', '대한수학회 공유확진자DB_220412-220418.csv', '범부처사업단 공유사업단 DB_220327(220325_).csv'])):
    
    data_test = region_preprocessing(path=files)
    #state_data, state_list 초기값 세팅
    if i == 0:
        # data_test = region_preprocessing(path=files)
        state_data, state_list = load_data_and_set_date_state(data_test, start_date, end_date)
        city_data, city_list = load_data_and_set_date_city(data_test, start_date, end_date)
        date_df = make_dummy_date(start_date, end_date)
        # print(city_list)
        # print(len(city_list))
        # input('ctrl + c')
        
    else:
        state_data, _ = load_data_and_set_date_state(data_test, start_date, end_date)
        city_data, _ = load_data_and_set_date_city(data_test, start_date, end_date)
    #     print(city_list)
    #     print(len(city_list))
    #     # input('ctrl + c')
    #     # print(city_data.head())
    # # # 각 파일을 state 별로 저장
    daily_df_state = sort_each_region_df(state_data, state_list, date_df, data_test, [], "state")
    daily_df_state.to_csv(f'Processing_data/{files[:-4]}_state.csv', encoding="euc-kr")
    # # # 각 파일을 city 별로 저장
    daily_df_city = sort_each_region_df(city_data, city_list, date_df, data_test, [], "city")
    daily_df_city.to_csv(f'Processing_data/{files[:-4]}_city.csv', encoding="euc-kr")
# input("input : ctrl + c")
# print(os.getcwd())
# 저장된 각 파일을 합쳐주는 부분
final_df_state = merge_each_df(date_df = date_df, region_list = state_list, path = './Processing_data', category='state')
final_df_city = merge_each_df(date_df = date_df, region_list = city_list, path = './Processing_data', category='city')
# final_df_city.to_csv('test.csv', encoding='cp949')
# print(final_df_city.head())
# input("Confirm Final Merge City Data : ")

# df 를 conf_data_state 로 수정해야함.
# final_df_state = pd.read_csv('Processing_Results/conf_case_state_test.csv', encoding='cp949', index_col=0)
# smoothing_and_save(df = final_df_state, smoothing_window=1, path='./Processing_Results', category='state_mean', df_save=True, fig_save=True)
# smoothing_and_save(df = final_df_city, smoothing_window=1, path='./Processing_Results', category='city_mean', df_save=True, fig_save=False)

# smoothing_and_save(df = final_df_state, smoothing_window=3, path='./Processing_Results', category='state_mean', df_save=True, fig_save=True)
# smoothing_and_save(df = final_df_city, smoothing_window=3, path='./Processing_Results', category='city_mean', df_save=True, fig_save=False)

# smoothing_and_save(df = final_df_state, smoothing_window=5, path='./Processing_Results', category='state_mean', df_save=True, fig_save=True)
# smoothing_and_save(df = final_df_city, smoothing_window=5, path='./Processing_Results', category='city_mean', df_save=True, fig_save=False)

# smoothing_and_save(df = final_df_state, smoothing_window=7, path='./Processing_Results', category='state_mean', df_save=True, fig_save=True)
# smoothing_and_save(df = final_df_city, smoothing_window=7, path='./Processing_Results', category='city_mean', df_save=True, fig_save=False)

smoothing_and_save(df = final_df_state, smoothing_window=10, path='./Processing_Results', category='state_mean', df_save=True, fig_save=True)
smoothing_and_save(df = final_df_city, smoothing_window=10, path='./Processing_Results', category='city_mean', df_save=True, fig_save=False)

smoothing_and_save(df = final_df_state, smoothing_window=20, path='./Processing_Results', category='state_mean', df_save=True, fig_save=True)
smoothing_and_save(df = final_df_city, smoothing_window=20, path='./Processing_Results', category='city_mean', df_save=True, fig_save=False)

smoothing_and_save(df = final_df_state, smoothing_window=30, path='./Processing_Results', category='state_mean', df_save=True, fig_save=True)
smoothing_and_save(df = final_df_city, smoothing_window=30, path='./Processing_Results', category='city_mean', df_save=True, fig_save=False)