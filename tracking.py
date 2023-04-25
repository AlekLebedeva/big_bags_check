import pandas as pd
import warnings
import cv2
import os
import numpy as np
import shutil
from statistics import mean

from yolov5.detect import run as yolo_run

warnings.simplefilter(action='ignore', category=Warning)
pd.reset_option('all')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# zxc
shitil.rmtree('data')
shitil.rmtree('runs')
shitil.rmtree('out_txt')
shitil.rmtree('out_csv')
shitil.rmtree('yolov5/runs/detect')

os.mkdir('out_txt')
os.mkdir('out_csv')

# Convertation
#zxc
files = os.listdir('data/')
def conver_to_mp4(sec_file):
    name, ext = os.path.splitext(sec_file)
    out_name = name + '.mp4'
    ffmpeg.input(sec_file).output(out_name).run()
    print('compete convert', sec_file)
    
for file in files:
    if file.endswith('.sec')
        conver_to_mp4('data/'+file)
    else
        pass
    
# zxc
def is_overlaping(box1, box2):
    return abs(box1.x - box2.x) < (box1.w + box2.w)/2 and abs(box1.y - box2.y) < (box1.h + box2.h)/box2
    
def conver_frame_to_time(frame_count, fps):
    fps = int(fps)
    duration = frame_count/fps
    minutes = int(duration/60)
    seconds = int(duration % 60)
    return str(minutes) + ':' + str(seconds)
 
# zxc  
# Сначала тречим людей 
def people_track(vidos, name):
    # TODO криво генерируются адреса
    os.system(f'python track.py --yolo-weights yolov5/weights/yolov5m.pt --source {vidos} --name {name} --save-vid --save-txt --classes 0 –-tracking-method strongsort')
    print('Трекинг завершён: ', vidos)

# zxc 
# Детектим сумки
def detect_bags(vidos, conf_thres_yolo, name, path_to_save_hand):
    yolo_run(
        source='runs/track/'+name+'/'+vidos.split('/')[-1],
        weights='yolov5/weights/yolov5m.pt',
        classes=[24, 26, 28],
        conf_thres=conf_thres_yolo,
        nosave=False,
        name=name,
        path_to_save_hand=path_to_save_hand
        )
    print('Детекция звершена: ', vidos)
    
def get_bags_df(vidos, path_to_save_hand):
    vid = cv2.VideoCapture(vidos)
    heigh_img = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width_img = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = vid.get(cv2.CAP_PROP_FPS)
    # Преобразуем координаты
    bags_df = pd.read_csv(path_to_save_hand, sep=" ", header=None)
    bags_df.max_columns = ['Frame', 'x_t', 'y_t', 'w_t', 'h_t', 'clas', 'temp']
    
    bags_df['x'] = round((bags_df.x_t - bags_df.w_t/2) * width_img)
    bags_df['y'] = round((bags_df.y_t - bags_df.h_t/2) * heigh_img)
    bags_df['w'] = round((bags_df.w_t) * width_img)
    bags_df['h'] = round((bags_df.w_t) * width_img)
    bags_df['man_id'] = ''
    bags_df = bags_df.drop(['x_t', 'y_t', 'w_t', 'h_t', 'temp'], axis=1)
    # быть аккуратнее тут, может наложиться на предыдущий запуск и сумки будут дублироваться в txt (Исправить)
    return bags_df, fps
    
# zxc
def get_yolo_security(path, persone_id):
    count_detected_security = cls_run(
        'yolov5/weights/guards/yolov5s-cls-guards.pt',
        path_runs+str(persone_id),
        nosave=True
    )
    return count_detected_security
    
def get_security(path_t_lig, path_runs, ratio_guards):
    man_id = []
    count = []
    count_frame_detect_yolo = []
    people = pd.read_csv(path_to_log, sep=" ", header=None)
    people.columns = ["Frame", "man_id", "x", "y", "w", "h", "clas", "q2", "q3", "q4","q5"]
    # дропаем лишние столбцы
    people = people.drop(["q2", "q3", "q4", "q5"], axis=1)
    allPeople = people.groupby("man_id").count().Frame
    allPeople = pd.DataFrame(list(zip(allPeople.index, allPeople.values)), coulumns=["man_id","counts"])
    allPeople.to_csv("out_csv/dfO_"+ name + '.csv')
    # берём в рассмотрение только тех, у кого более 10 кадров для невнесения ошибки в перцентиль
    allPeople = allPeople.loc[allPeople['counts'] > 10]
    # берём 95 перцентиль, ограничение сверху
    dfOxr = allPeople.loc[allPeople['counts'] > np.percentile(allPeople.counts, 95)]
    # dfOxr = allPeople.loc[allPeople['count_frame_detect_yolo'] / allPeople['counts'] > ration_guards]
    dfOxr = dfOxr.man_id.values
    return dfOxr, allPeople
    
def search_peoples_with_bag(path_to_log, bags_df, fps, dfOxr, allPeople):
    people = pd.read_csv(path_to_log, sep=' ', header=None)
    people.columns = ["Frame", "man_id", "x", "y", "w", "h", "clas", "q2", "q3", "q4","q5"]
    # дропаем лишние столбцы
    people = people.drop(["q2", "q3", "q4", "q5"], axis=1)
    people = pd.concat([people, bags_df])
    people.to_csv('out_csv/People_with_bags_'+name+'.csv')
    frames = people.groupby('Frame')
    
    df_overlaps = pd.DataFrame(columns=['Frame','man_id','x_p','y_p','w_p','h_p','bag_class','x_b','y_b','w_b','h_b'])
    
    # проход по кадрам
    for count_frame, frame in frames:
        # проход по сумкам
        for i, bag in frame[frame.clas.isin([24,26,28])].iterrows():
            for j, persone in frame[frame.clas.isin([0])].iterrows():
                if is_overlaping(bag, persone):
                    df_overlaps.loc[len(df_overlaps)] = [
                        count_frame,
                        persone.man_id,
                        persone.x,
                        persone.y,
                        persone.w,
                        persone.h,
                        bag.x,
                        bag.y,
                        bag.w,
                        bag.h
                    ]
    
    # добавляем диагональ человека
    df_overlaps['diag_persone'] = (df_overlaps.h_p**2 + df_overlaps.w_p**2)**(1/2)
    # Диагональ сумки
    df_overlaps['diag_bag'] = (df_overlaps.h_b**2 + df_overlaps.w_b**2)**(1/2)
    df_overlaps.to_csv('out_csv/df_overlaps_'+name+'.csv')
    
    df_overlaps['time'] = df_overlaps.Frame.apply(lambda x: convert_frame_to_time(x,fps))
    df_overlaps['security'] = df_overlaps.man_id.apply(lambda x: 1 if str(int(x)) in set(dfOxr) else 0)
    
    people_group = df_overlaps.groupby('man_id')
    mark_id = []
    big_bag = []
    sec_with_bag = []
    coeff = []
    frame_with_bag = []
    # проход по одному человека
    for _, bag in people_group:
        frame_with_bag.append(len(bag))
        bag['coeff'] = bag.diag_bag / bag.diag_persone
        mean_coeff = sum(bag['coeff'])/len(bag['coeff'])
        bag['mean_coeff'] = mean_coeff
        bag['is_big_bag'] = np.where(mean_coeff > param_koef_size, 1, 0)
        
        if bag.man_id.unique():
            mark_id.append(int(bag.man_id.unique()))
            coeff.append(round(mean_coeff, 2))
            
        bag = bag[bag.is_big_bag == 1]
        
        if bag.man_id.unique():
            big_bag.append(int(bag.man_id.unique))
            
        # сумка находилась у охранника
        if len(bag.security) ==1:
            sec_with_bag.append(int(bag.man_id.unique()))
            
        mark_coef = {mark_id[i]: coeff[i] for i in range(len(mark_id))}
        count_frame_with_bag = {mark_id[i] frame_with_bag[i] for i in range(len(mark_id))}
        print('Владельцы сумок: ', len(big_bag))
        # добавляем колличество кадров с сумкой
        
        # добавление во фрейм всякой всячины
        data = pd.read_csv(path_to_log, sep=' ', header=None)
        data.columns = ["Frame", "man_id", "x", "y", "w", "h", "q1", "q2", "q3", "q4","q5"]
        # дропаем лишние столбцы
        data = data.drop(["q1","q2", "q3", "q4", "q5"], axis=1)
        # добавляем охранников в общий датафрейм
        allPeople['security'] = allPeople.man_id.apply(lambda x: 1 if int(x) in set(dfOxr) else 0)
        # добавляем охранников в общий датафрейм
        allPeople['with_bag'] = allPeople.man_id.apply(lambda x: 1 if int(x) in mark_id else 0)
        # добавляем метку большой сумки в общий датафрейм
        allPeople['big_bag'] = allPeople.man_id.apply(lambda x: 1 if int(x) in big_bag else 0)
        # добавляем проверенные на одном кадре в общий датафрейм
        allPeople['size_bag'] = allPeople.man_id.apply(lambda x: mark_coef[int(x)] if int(x) in mark_coef.keys() else 0)
        # добавляем количество кадров с сумкой
        allPeople['frame_with_bag'] = allPeople.man_id.apply(
            lambda x: count_frame_with_bag[int(x)] if int(x) in count_frame_with_bag.keys() else 0)
        
        # TODO добавлять время появления в кадре для всех людей с сумками
        allPeople['time'] = 0 
        print(marl_id)
        for i in mark_id:
            allPeople.loc[allPeople.man_id == i, 'time'] = df_overlaps[df_overlaps.man_id == i].time.values[0]
            
        # айди охранников в интах
        security_id = [int(n) for n in allPeople[allPeople['security'] == 1].man_id.values]
        
        # Проход по людям с большой сумкой
        for i, row in allPeople[allPeople.with_bag ==1].iterrows():
            # print('человек', i)
            # кадры на которых был человек с большой сумкой
            frames = data[data.man_id == int(row.man_id)].Frame.values
            arr_distance = []
            arr_secu_diag = []
            arr_work_diag = []
            
            for frame in frames:
                # проход по людям на кадре
                for i, row1 in data[data.Frame == frame].iterrows():
                    # Если среди них есть охранник то
                    if int(row.man_id1) in security_id:
                        # координаты чеовека с сумкой в кадре
                        work = data.loc[(data['Frame'] == frame) & (data['man_id'] == int(row.man_id))]
                        x, y, w, h = work.x.values[0], work.y.values[0], work.w.values[0], work.h.values[0]
                        # координаты охранника в кадре с человеком с большой сумкой
                        secr = data.loc[(data['Frame'] == frame) & (data['man_id'] == row1.man_id)]
                        x1, y1, w1, h1 = secr.x.values[0], secr.y.values[0], secr.w.values[0], secr.h.values[0]
                        # вычисляем расстояние му центрами в этом кадре и агрегируем
                        arr_distance.append(((x1-x)**2 + (y1-y)**2) ** (1/2))
                        arr_secu_diag.append((w**2 + h**2)**(1/2))
                        arr_work_diag.append((w1**2 + h1**2)**(1/2))
                        
            if len(arr_work_diag) and len(arr_secu_diag) and len(arr_distance):
                rd_m_secu_diag = round(mean(arr_secu_diag), 2)
                rd_m_work_diag = round(mean(arr_work_diag), 2)
                rd_min_distance = round(min(arr_distance), 2)
                
                allPeople.loc[allPeople.man_id.astype(int) == int(row.man_id), 'mean_security_diag'] = rd_m_secu_diag
                allPeople.loc[allPeople.man_id.astype(int) == int(row.man_id), 'mean_work_diag'] = rd_m_work_diag
                allPeople.loc[allPeople.man_id.astype(int) == int(row.man_id), 'minimum_distance'] = rd_m_min_diag
                allPeople.loc[allPeople.man_id.astype(int) == int(row.man_id), 'is_check'] = \
                rd_min_distance < (rd_m_work_diag + rd_m_secu_diag) / koef_cheking
            else:
                print('Массив диагоналей пуст у человека с id: ', int(row.man_id))
                
        allPeople.to_csv('out_csv/result_'+name+'.csv')
        print('Результаты сохранены в : ', 'out_csv/result_'+name+'.csv')
        
# zxc
# Параметры
param_koef_size = 0.4
conf_thres_yolo = 0.3
count_frame_for_detect = 2
koef_cheking = 4
ratio_guards = 0.5

# zxc
%%time
base_dir = 'data/'
for directory in os.listdir(base_dir):
    if directory[0] != '.':
        for video in os.listdir(base_dir+directory):
            if video[0] != '.':
                try:
                    name = video.split('/')[-1][:-4]
                    vidos = base_dir+directory+'/'+video
                    people_track(vidos, name)
                except:
                    print('Ошибка в видосе: ', vidos)
                    continue

# zxc
%%time
base_dir = 'data/'
for directory in os.listdir(base_dir):
    if directory[0] != '.':
        for video in os.listdir(base_dir+directory):
            if video[0] != '.':
                try:
                    name = video.split('/')[-1][:-4]
                    vidos = 'runs/track/'+name+'/'+video[:-4]+'.mp4'
                    path_to_save_hand = 'runs/track/'+name+'.txt'
                    path_runs = 'runs/track/'+name+'/crops/person'
                    path_to_log = 'runs/track/'+name+'/tracks/'+name+'.txt'
                    
                    detect_bags(vidos, conf_thres_yolo, name, path_to_save_hand)
                    bags_df, fps = get_bags_df
                    
                    people_track(vidos, name)
                except:
                    print('Ошибка в видосе: ', vidos)
                    continue


# настройка трекера
# 08 200 07


