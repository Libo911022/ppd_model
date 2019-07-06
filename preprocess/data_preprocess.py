import pandas as pd
import numpy as np
import datetime as dt
from joblib import parallel, delayed
import logging
import pickle
from collections import Counter
import gc
def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger
logger = get_logger()

def run_parallel(grouped, method,verbose=True):
    # n_jobs=-1最大线程
    with parallel.Parallel(n_jobs=6,backend='multiprocessing',verbose=verbose) as par:
        segments = par(delayed(method)(df_seg) for df_seg in grouped)
    return segments

def user_info_features():
    logger.info('operating user info')
    user_info = pd.read_csv('../user_info.csv')
    user_info['reg_mon'] = user_info['reg_mon'].map(lambda x:dt.datetime.strptime(x+'-01','%Y-%m-%d'))
    user_info['insertdate'] = pd.to_datetime(user_info['insertdate'])

    user_info['insert_last'] = user_info.groupby('user_id')['insertdate'].transform(max)
    user_info['insert_first'] = user_info.groupby('user_id')['insertdate'].transform(min)
    user_info['cell_max'] = user_info.groupby('user_id')['cell_province'].transform(max)
    user_info['cell_min'] = user_info.groupby('user_id')['cell_province'].transform(min)

    user_info = user_info[user_info.insert_last == user_info.insertdate]
    user_info['max_interval'] = user_info.apply(lambda x:(x.insert_last-x.insert_first).days//30,axis=1)
    user_info['reg_interval'] = user_info.apply(lambda x:(x.insert_first-x.reg_mon).days//30,axis=1)
    user_info['cell_change'] = user_info.apply(lambda x: 1 if x.cell_max!=x.cell_min else 0,axis=1)
    user_info = user_info[['user_id', 'gender', 'age', 'cell_province', 'id_province','id_city','max_interval','reg_interval','cell_change']].set_index('user_id')
    with open('user_info_features.pkl','wb') as file:
        pickle.dump(user_info,file)
    # return user_info

def user_taglist_features():
    logger.info('operating user_taglist')
    user_taglist = pd.read_csv('../user_taglist.csv')
    user_taglist = user_taglist.groupby('user_id').agg({'insertdate': [('tag_cnt', 'count')],'taglist':list})
    user_taglist['taglist'] = user_taglist.apply(lambda x:list(set('|'.join([y[0] for y in x.taglist]).split('|'))) if len(x.taglist)>0 else [],axis=1)
    with open('user_taglist_features.pkl','wb') as file:
        pickle.dump(user_taglist,file)
    # return user_taglist

def listing_operator(df):
    df.sort_values('auditing_date',inplace=True)
    latest_day = df.auditing_date.max()
    df['days_diff'] = df['auditing_date'].map(lambda x:(latest_day-x).days)
    max_interval = df['days_diff'].max()
    feature_dict = {'user_id':df.user_id.values[0]}
    time_mapping = {180:'_6m',360:'_12m',720:'_24m'}
    if max_interval<=180:
        time_list = [180]
    elif max_interval<=360:
        time_list = [180,360]
    else:
        time_list = [180, 360,720]
    for t in time_list:
        prefix = time_mapping[t]
        temp = np.array(df.query('days_diff<=%s'%(t))[['term','rate','principal','interset_per','auditing_date']].values)
        if len(temp[:,0])>0:
            #申请标的次数
            col_name = 'listing_num'+prefix
            feature_dict[col_name] = len(temp[:,0])
            #最大，最小，平均期数
            col_name = 'max_term'+prefix
            feature_dict[col_name] = np.max(temp[:,0])
            col_name = 'mean_term' + prefix
            feature_dict[col_name] = np.mean(temp[:, 0])
            col_name = 'min_term' + prefix
            feature_dict[col_name] = np.min(temp[:, 0])

            #最大，最小，平均利率
            col_name = 'max_rate'+prefix
            feature_dict[col_name] = np.max(temp[:,1])
            col_name = 'mean_rate' + prefix
            feature_dict[col_name] = np.mean(temp[:, 1])
            col_name = 'min_rate' + prefix
            feature_dict[col_name] = np.min(temp[:, 1])

            #最大，最小，平均总和金额
            col_name = 'max_principal'+prefix
            feature_dict[col_name] = np.max(temp[:,2])
            col_name = 'mean_principal' + prefix
            feature_dict[col_name] = np.mean(temp[:, 2])
            col_name = 'min_principal' + prefix
            feature_dict[col_name] = np.min(temp[:, 2])
            col_name = 'sum_principal' + prefix
            feature_dict[col_name] = np.sum(temp[:, 2])

            #每期利息
            col_name = 'max_interset_per'+prefix
            feature_dict[col_name] = np.max(temp[:,3])
            col_name = 'mean_interset_per' + prefix
            feature_dict[col_name] = np.mean(temp[:, 3])
            col_name = 'min_interset_per' + prefix
            feature_dict[col_name] = np.min(temp[:, 3])
            col_name = 'sum_interset_per' + prefix
            feature_dict[col_name] = np.sum(temp[:, 3])

            #单日最多申请标的
            col_name = 'max_listing_per_day' + prefix
            feature_dict[col_name] = max(Counter(temp[:, 4]).values())

            #单月最多申请标的
            col_name = 'max_listing_per_month' + prefix
            feature_dict[col_name] = max(Counter([x.year*1000+x.month for x in temp[:, 4]]).values())
    return feature_dict

def listing_info_features():
    logger.info('operating listing_info')
    listing_info = pd.read_csv('../listing_info.csv')
    listing_info['auditing_date'] = pd.to_datetime(listing_info['auditing_date'])
    listing_info['interset_per'] = listing_info['principal']*listing_info['rate']/listing_info['term']
    list_df = [d[1].copy() for d in listing_info.groupby('user_id')]
    logger.info("start calculating listing features")
    agg_features = []
    grouped = run_parallel(list_df, listing_operator)
    grouped = (x for x in grouped)
    for data in grouped:
        agg_features.append(data)
    final = pd.DataFrame(agg_features)
    listing_info = listing_info.merge(final,on='user_id',how='left')
    with open('listing_info_features.pkl','wb') as file:
        pickle.dump(listing_info,file)

def behavior_operator(df):
    df['behavior_month'] = df['behavior_time'].map(lambda x: x.year * 100 + x.month)
    df['behavior_hour'] = df['behavior_time'].map(lambda x: x.hour)
    df.sort_values('behavior_month', inplace=True)
    month_mapping = dict()
    month_list = list(df['behavior_month'].unique())
    for x, y in enumerate(month_list[::-1]):
        month_mapping[y] = x
    df['behavior_month'] = df['behavior_month'].map(month_mapping)
    final_dict = {'user_id':df.user_id.values[0]}
    
    #用户连续行为
    if len(month_list)>1:
        month_list_diff = [month_list[i] - month_list[i - 1] for i in range(1, len(month_list))]
        month_list_diff = [x if x < 12 else x - 88 for x in month_list_diff]
        col_name = 'max_month_diff'
        final_dict[col_name] = np.max(month_list_diff)
        col_name = 'min_month_diff'
        final_dict[col_name] = np.min(month_list_diff)
        col_name = 'mean_month_diff'
        final_dict[col_name] = np.mean(month_list_diff)
        col_name = 'sum_month_diff'
        final_dict[col_name] = np.sum(month_list_diff)
    else:
        final_dict['max_month_diff'] = 0
        final_dict['min_month_diff'] = 0
        final_dict['mean_month_diff'] = 0
        final_dict['sum_month_diff'] = 0
          
    for n_month in [1, 3, 6, 12]:
        for btype in [[1], [2, 3]]:
            prefix = 'last_%s_bhmonth_type%s' % (n_month, btype[0]) if btype != [2, 3] else 'last_%s_bhmonth_type23' % (n_month)
            temp = df[(df.behavior_month<=n_month)&(df.behavior_type.isin(btype))]
            if len(temp)>0:
                col_name = prefix + '_cnt'
                final_dict[col_name] = len(temp)
                #同一时间内最多点击次数
                col_name = prefix + '_max_click_cnt_day'
                final_dict[col_name] = temp['behavior_time'].value_counts().values[0]

                #日内点击行为统计
                day_agg = temp.groupby('behavior_day').size().values.tolist()
                if len(day_agg) > 0:
                    col_name = prefix + '_max_cnt_day'
                    final_dict[col_name] = np.max(day_agg)
                    col_name = prefix + '_mean_cnt_day'
                    final_dict[col_name] = np.mean(day_agg)
                    col_name = prefix + '_min_cnt_day'
                    final_dict[col_name] = np.min(day_agg)
                    col_name = prefix + '_std_cnt_day'
                    final_dict[col_name] = np.std(day_agg)
                else:
                    final_dict[prefix + '_max_cnt_day'] = 0
                    final_dict[prefix + '_mean_cnt_day'] = 0
                    final_dict[prefix + '_min_cnt_day'] = 0
                    final_dict[prefix + '_std_cnt_day'] = 0
                 #凌晨1点到7点点击行为统计
                dawn_behavior = temp.query("behavior_hour>=1 and behavior_hour<=7")
                col_name = prefix + '_dawn_click_pct'#凌晨点击占比
                final_dict[col_name] = len(dawn_behavior)*1.0/len(temp)
                dawn_behavior_agg = dawn_behavior.groupby('behavior_day').size().values.tolist()
                if len(dawn_behavior_agg)>0:
                    col_name = prefix + '_max_cnt_day_dawn'
                    final_dict[col_name] = np.max(dawn_behavior_agg)
                    col_name = prefix + '_mean_cnt_day_dawn'
                    final_dict[col_name] = np.mean(dawn_behavior_agg)
                    col_name = prefix + '_std_cnt_day_dawn'
                    final_dict[col_name] = np.std(dawn_behavior_agg)
                else:
                    final_dict[prefix + '_max_cnt_day_dawn'] = 0
                    final_dict[prefix + '_mean_cnt_day_dawn'] = 0
                    final_dict[prefix + '_std_cnt_day_dawn'] = 0
                #一小时内平均点击最大，方差
                dawn_behavior_hour_agg = dawn_behavior.groupby(['behavior_day','behavior_hour']).size().values.tolist()
                if len(dawn_behavior_hour_agg)>0:
                    col_name = prefix + '_max_cnt_hour'
                    final_dict[col_name] = np.max(dawn_behavior_hour_agg)
                    col_name = prefix + '_mean_cnt_hour'
                    final_dict[col_name] = np.mean(dawn_behavior_hour_agg)
                    col_name = prefix + '_std_cnt_hour'
                    final_dict[col_name] = np.std(dawn_behavior_hour_agg)
                else:
                    final_dict[prefix + '_max_cnt_hour'] = 0
                    final_dict[prefix + '_mean_cnt_hour'] = 0
                    final_dict[prefix + '_std_cnt_hour'] = 0
            else:
                final_dict[prefix + '_max_click_cnt_day'] = 0
                final_dict[prefix + '_cnt'] = 0
    return final_dict

def user_behavior_features():
    behavior = pd.read_csv("../user_behavior_logs.csv")
    behavior['behavior_day'] = behavior['behavior_time'].str[:10]
    behavior['behavior_time'] = pd.to_datetime(behavior['behavior_time'])
    logger.info("start calculating behavior features")
    list_df = [d[1].copy() for d in behavior.groupby('user_id')]
    agg_features = []
    grouped = run_parallel(list_df, behavior_operator)
    grouped = (x for x in grouped)
    for data in grouped:
        agg_features.append(data)
    final = pd.DataFrame(agg_features)
    logger.info("finish  behavior features")
    with open('behavior_features.pkl','wb') as file:
        pickle.dump(final,file)

def user_repay_operator(df):
    feature_dict = {'user_id':df.user_id.values[0]}
    df['prepay_days'] = df.apply(lambda x:(x.due_date-x.repay_date).days,axis=1)
    df['prepay_days'] = df['prepay_days'].map(lambda x: x if x >= 0 else -1)    
    prepay_days = np.array(df.prepay_days.values.tolist())
    #提前还款次数(占比)
    col_name = 'prepay_time'
    feature_dict[col_name] = np.sum(prepay_days>0)
    col_name = 'prepay_time_pct'
    feature_dict[col_name] = np.sum(prepay_days > 0)/len(prepay_days)
    
    #提前连续还款期数平均（大于1期）,最大
    col_name = 'max_continue_repay_times_once'
    feature_dict[col_name] = np.max([round(x/30) for x in prepay_days])
    col_name = 'mean_coninue_repay_times_once'
    feature_dict[col_name] = np.mean([round(x/30) for x in prepay_days if round(x/30)>1])
        
    #统计历史提前还款频率，大于31天按31天算
    df['prepay_days'] = df['prepay_days'].map(lambda x: x if x<=31 else 31)
    prepay_days = np.array([x if x<=31 else 31 for x in prepay_days])
    prepay_vc = Counter(prepay_days)
    df['amt_weight'] = df['prepay_days'].map(lambda x:prepay_vc[x]/len(prepay_days))
    df['weighted_amt'] = df.apply(lambda x: round(x.amt_weight*x.due_amt,4) ,axis=1)
    weighted_amt = df.weighted_amt.values.tolist()

    #提前还款天数,金额平均，最大，众数，
    col_name = 'prepay_days_max'
    feature_dict[col_name] = np.max(prepay_days)
    col_name = 'prepay_days_mean'
    feature_dict[col_name] = np.mean(prepay_days)
    col_name = 'prepay_days_mode'
    feature_dict[col_name] = max(prepay_vc,key=prepay_vc.get)
    
    if len(weighted_amt)>0:
        col_name = 'weighted_amt_max'
        feature_dict[col_name] = np.max(weighted_amt)
        col_name = 'weighted_amt_mean'
        feature_dict[col_name] = np.mean(weighted_amt)
        col_name = 'weighted_amt_min'
        feature_dict[col_name] = np.min(weighted_amt)
    else:
        feature_dict['weighted_amt_max'] = 0
        feature_dict['weighted_amt_mean'] = 0
        feature_dict['weighted_amt_min'] = 0
        
    #逾期次数(占比)
    col_name = 'default_times'
    feature_dict[col_name] = np.sum(prepay_days < 0)
    col_name = 'default_times_pct'
    feature_dict[col_name] = np.sum(prepay_days < 0)/len(prepay_days)

    #逾期金额
    default_amt = df.query("prepay_days<0").due_amt.values.tolist()
    if len(default_amt)>0:
        col_name = 'default_amt_max'
        feature_dict[col_name] = np.max(default_amt)
        col_name = 'default_amt_mean'
        feature_dict[col_name] = np.mean(default_amt)
        col_name = 'default_amt_min'
        feature_dict[col_name] = np.min(default_amt)
    else:
        feature_dict['default_amt_max'] = 0
        feature_dict['default_amt_mean'] = 0
        feature_dict['default_amt_min'] = 0

    #正常还款次数（占比）
    col_name = 'normal_times'
    feature_dict[col_name] = np.sum(prepay_days==0)
    col_name = 'normal_times_pct'
    feature_dict[col_name] = np.sum(prepay_days == 0)/len(prepay_days)

    #正常还款金额
    normal_amt = df.query("prepay_days==0").due_amt.values.tolist()
    if len(normal_amt)>0:
        col_name = 'normal_amt_max'
        feature_dict[col_name] = np.max(normal_amt)
        col_name = 'normal_amt_mean'
        feature_dict[col_name] = np.mean(normal_amt)
        col_name = 'normal_amt_min'
        feature_dict[col_name] = np.min(normal_amt)
    else:
        feature_dict['normal_amt_max'] = 0
        feature_dict['normal_amt_mean'] = 0
        feature_dict['normal_amt_min'] = 0

    first_order = df.query("order_id==1")
    if first_order.shape[0]>0:
        #第一期提前还款天数平均，最大,众数
        first_prepay_list = first_order.prepay_days.values.tolist()
        first_prepay_vc = Counter(first_prepay_list)
        col_name = 'first_prepay_days_max'
        feature_dict[col_name] = np.max(first_prepay_list)
        col_name = 'first_prepay_days_mean'
        feature_dict[col_name] = np.mean(first_prepay_list)
        col_name = 'first_prepay_days_mode'
        feature_dict[col_name] = max(first_prepay_vc,key=first_prepay_vc.get)
        #第一期提前还款金额平均，最大
        first_prepay_amt_list = first_order.due_amt.values.tolist()
        col_name = 'first_prepay_due_amt_max'
        feature_dict[col_name] = np.max(first_prepay_amt_list)
        col_name = 'first_prepay_due_amt_min'
        feature_dict[col_name] = np.min(first_prepay_amt_list)
        col_name = 'first_prepay_due_amt_mean'
        feature_dict[col_name] = np.mean(first_prepay_amt_list)
    else:
        feature_dict['first_prepay_days_max'] = 0
        feature_dict['first_prepay_days_min'] = 0
        feature_dict['first_prepay_days_mode'] = 0
        feature_dict['first_prepay_due_amt_max'] = 0
        feature_dict['first_prepay_due_amt_min'] = 0
    return feature_dict

def user_repay_features():
    #数据观察发现，部分标的的还款行为并没有提供所有账单，所以只能通过统计信息挖掘用户的还款习惯
    #训练集和测试集有部分用户id并没有记录还款记录，对于训练集可以添加第一期结果到还款表中，测试集部分用户只能通过标的基本信息完成
    user_repay_logs = pd.read_csv('../user_repay_logs.csv')
    train_data = pd.read_csv('../train.csv')
    train_data = train_data[~train_data.listing_id.isin(user_repay_logs.user_id.unique())]
    train_data['repay_amt'] = train_data['due_amt']
    train_data['repay_date'] = train_data['repay_date'].map(lambda x: '2200-01-01' if x == '\\N' else x)
    train_data.drop('auditing_date', axis=1, inplace=True)
    train_data['order_id'] = 1
    user_repay_logs = pd.concat([user_repay_logs,train_data])

    user_repay_logs['due_date'] = pd.to_datetime(user_repay_logs['due_date'])
    user_repay_logs['repay_date'] = pd.to_datetime(user_repay_logs['repay_date'])
    list_df = [d[1].copy() for d in user_repay_logs.groupby('user_id')]
    logger.info("start calculating user repay features")
    agg_features = []
    grouped = run_parallel(list_df, user_repay_operator)
    gc.collect()
    grouped = (x for x in grouped)
    for data in grouped:
        agg_features.append(data)
    final = pd.DataFrame(agg_features)
    with open('user_repay_features.pkl','wb') as file:
        pickle.dump(final,file)

if __name__ == '__main__':
   user_info_features()
   user_taglist_features()
   user_behavior_features()
   user_repay_features()
   listing_info_features()
