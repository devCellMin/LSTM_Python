# GUI
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as msgbox
from tkinter import *  # __all__
from tkinter import filedialog

# Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import FinanceDataReader as fdr

# Model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback

# ETC
import os
import warnings

'''Code 시작'''
root = Tk()
root.title("주가예측 머신러닝")  # 제목
root.geometry("360x640")  # GUI 화면 사이즈

'''global Value(Hyper parameter)'''
STOCK_CODE = 0
WINDOW_SIZE = 20
BATCH_SIZE = 32  # 원래는 32
Epoch_num = 50
while_TF_early_stop = True
# pred, y_test, stock also global

'''dataset 함수'''


def windowed_dataset(series, window_size, batch_size, shuffle):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    return ds.batch(batch_size).prefetch(1)


'''GUI 함수 코드'''


def get_fincode():
    code = entry_add_code.get()
    return code


def read_fin_history():
    global STOCK_CODE
    global stock
    STOCK_CODE = get_fincode()
    terminal_listbox.insert(
        END, "입력받은 주식코드({})의 데이터를 가져옵니다." .format(STOCK_CODE))
    stock = fdr.DataReader(STOCK_CODE)
    terminal_listbox.insert(END, "성공하였습니다.")


def show_graph():
    global pred
    global y_test
    plt.figure(figsize=(12, 9))
    plt.plot(np.asarray(y_test)[20:], label='actual')
    plt.plot(pred, label='prediction')
    plt.legend()
    plt.show()


def start_Learning():  # 머신러닝 시작
    warnings.filterwarnings('ignore')
    global while_TF_early_stop
    while_TF_early_stop = True

    global WINDOW_SIZE
    global BATCH_SIZE
    global pred
    global y_test
    global stock
    global inverse_max_min_scaler

    # 데이터 전처리 시작
    scaler = MinMaxScaler()
    # 스케일을 적용할 column을 정의합니다.
    scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    # 스케일 후 columns
    scaled = scaler.fit_transform(stock[scale_cols])

    df = pd.DataFrame(scaled, columns=scale_cols)

    x_train, x_test, y_train, y_test = train_test_split(
        df.drop('Close', 1), df['Close'], test_size=0.2, random_state=0, shuffle=False)

    # trian_data는 학습용 데이터셋, test_data는 검증용 데이터셋 입니다.
    train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
    test_data = windowed_dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)

    model = Sequential([
        # 1차원 feature map 생성
        Conv1D(filters=48, kernel_size=5,
               padding="causal",
               activation="relu",
               input_shape=[WINDOW_SIZE, 1]),
        # LSTM
        LSTM(16, activation='tanh'),
        Dense(16, activation="relu"),
        Dense(1),
    ])

    # Sequence 학습에 비교적 좋은 퍼포먼스를 내는 Huber()를 사용합니다.
    loss = Huber()
    optimizer = Adam(0.0005)
    model.compile(loss=Huber(), optimizer=optimizer,
                  metrics=['mse'])

    # val_loss 기준 체크포인터도 생성합니다.
    filename = os.path.join('tmp', 'ckeckpointer.ckpt')
    checkpoint = ModelCheckpoint(filename,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_loss',
                                 verbose=1)

    global Epoch_num  # Epoch 횟수를 golbal로 가져온다.
    # Train data의 수에 Batch_size를 나누어서 Iteration을 구한다.
    train_data_count = x_train.shape[0]
    Iteration = (train_data_count//BATCH_SIZE)

    class calculate_Iteration(Callback):
        step = 0

        def plus_step(self):
            self.step += 1

    class print_log(calculate_Iteration):
        def __init__(self, Iteration):
            self.epoch_count = 0
            self.in_count = 0
            self.stop_count = 0
            self.Iteration = Iteration
            self.metric_cache = {}

        def print_history(self, step, dic_history={}):
            metrics_log = ''
            global Epoch_num
            self.step = step
            if(self.in_count < self.epoch_count):
                for key, value in dic_history.items():
                    self.metric_cache[key] = value
                for (k, v) in self.metric_cache.items():
                    val = v[-1] / 1000
                    if abs(val) > 1e-3:
                        metrics_log += ' - %s: %.4f' % (k, val)
                    else:
                        metrics_log += ' - %s: %.4e' % (k, val)
                terminal_listbox.insert(END, 'Epoch: {:4}/{} ... Iteration: {:4}/{}\n{}'.format(self.epoch_count, Epoch_num, step,
                                                                                                self.Iteration,
                                                                                                metrics_log))
                self.in_count += 1

        # earlystopping은 15번 epoch통안 val_loss 개선이 없다면 학습을 멈춥니다.
        def early_stoping(self, arr_history=[]):
            global while_TF_early_stop
            if(len(arr_history) >= 2):
                # arr_history가 2차원 배열의 형태를 띄고 있어서 [][]로 표현
                prev = str(arr_history[-2][0])
                now = str(arr_history[-1][0])
                float(prev)*0.01 > float(prev) - float(now)
                self.stop_count += 1
                # val_loss의 개선이 1퍼센트 이상으로 15번 개선이 없다면 학습을 멈춘다.
                if(self.stop_count >= 15):
                    while_TF_early_stop = False
                del(arr_history)

    cal_step = calculate_Iteration()
    out_batch = print_log(Iteration)
    val_loss_history = []
    count = 0
    while (count < Epoch_num) and (while_TF_early_stop):
        history = model.fit(train_data,
                            validation_data=(test_data),
                            epochs=1,
                            callbacks=[cal_step],
                            verbose=1)
        out_batch.epoch_count += 1  # Epoch의 반복횟수
        # 1회 callback을 통하여 step을 받아서 밑에 print_history에 넣어준다.
        step = cal_step.step
        terminal_listbox.insert(
            END, out_batch.print_history(step, history.history))
        val_loss_history.append(history.history['val_loss'])
        out_batch.early_stoping(val_loss_history)
        cal_step.step = 0
        count += 1
    model.load_weights(filename)
    pred = model.predict(test_data)
    # 최근의 예상값을 출력합니다.
    maxim = stock.max()
    minim = stock.min()
    range = maxim['Close']-minim['Close']
    result = pred * range + minim['Close']
    pred_price = result[-1][0]
    terminal_listbox.insert(END, "예상금액 : {}" .format(pred_price))


def start():  # 시작
    terminal_listbox.insert(END, "가져온 데이터를 통하여 러닝을 시작합니다.")
    start_Learning()


# 파일 프레임 (파일 추가, 선택 삭제)
file_frame = Frame(root)
file_frame.pack(fill="x", padx=5, pady=5)  # 간격 띄우기

add_code_label = Label(file_frame, text="주식코드 입력칸").pack()
entry_add_code = Entry(file_frame, width=30)
entry_add_code.place(height=30)
entry_add_code.pack(side="left", padx=5, pady=5)
btn_read_data = Button(file_frame, padx=5, pady=5, width=12,
                       text="데이터 읽어오기", command=read_fin_history).pack(side="right")

# 터미널 내용 넣기
terminal = Frame(root, bg='black')
terminal_scrollbar = tk.Scrollbar(terminal)
terminal_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
terminal_listbox = tk.Listbox(terminal, bg='white', fg='black', highlightcolor='white',
                              highlightthickness=0, selectbackground='black', activestyle=tk.NONE)
terminal_listbox.pack(expand=True, fill=tk.BOTH)
terminal.pack(expand=True, fill=tk.BOTH)

# Inserting the copyright thingy.
terminal_listbox.insert(tk.END, 'Tkinter Terminal')
terminal_listbox.insert(
    tk.END, "BATCH_SIZE : {} / Epoch : {}" .format(BATCH_SIZE, Epoch_num))
terminal_listbox.insert(tk.END, "# 네이버 : 035420 / 카카오 : 035720")
terminal_listbox.insert(
    tk.END, "# GOOGLE(A_Class) : GOOGL / GOOGLE(C_Class) : GOOG")
terminal_text = '>> '

# Assigns a scrollbar to the terminal.
terminal_listbox.config(yscrollcommand=terminal_scrollbar.set)
terminal_scrollbar.config(command=terminal_listbox.yview)


# 실행 프레임
frame_run = Frame(root)
frame_run.pack(fill="x", padx=5, pady=5)

btn_close = Button(frame_run, padx=5, pady=5, text="닫기",
                   width=12, command=root.quit).pack(side="right", padx=5, pady=5)
btn_start = Button(frame_run, padx=5, pady=5, text="표 출력",
                   width=12, command=show_graph).pack(side="right", padx=5, pady=5)
btn_start = Button(frame_run, padx=5, pady=5, text="시작",
                   width=12, command=start).pack(side="right", padx=5, pady=5)


root.resizable(True, True)  # x(너비), y(너비) 값 변경 불가 (창 크기 변경 불가)
root, mainloop()
