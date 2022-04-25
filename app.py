# You can write code above the if-main block.
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.callbacks import EarlyStopping, ModelCheckpoint

def norm(data):
    data_norm = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return data_norm

def reNorm(data, min_, diff_):
    return np.round(data[:] * diff_ + min_, decimals=2)

def setXY(data, ref_day, predict_day):
    x, y = [], []
    for i in range(len(data) - ref_day - predict_day):
        x.append(np.array(data.iloc[i:i + ref_day]))
        y.append(np.array(data.iloc[i + ref_day:i + ref_day + predict_day]['open']))
    x,y = np.array(x), np.array(y)
    return x, y

def buildLSTM(input_shape, output_shape):
    model = Sequential()
    model.add(LSTM(units=900, return_sequences = True, kernel_initializer = 'glorot_uniform', input_shape = (input_shape[1], input_shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(units = 900, kernel_initializer = 'glorot_uniform', return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 300, kernel_initializer = 'glorot_uniform', return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 300, kernel_initializer = 'glorot_uniform', return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 300, kernel_initializer = 'glorot_uniform', return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 300, kernel_initializer = 'glorot_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(units = output_shape))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mse'])
    model.summary()
    
    return model

def shuffle(x, y):
    np.random.seed(int(time.time()))
    randomList = np.arange(x.shape[0])
    np.random.shuffle(randomList)
    return x[randomList], y[randomList]

if __name__ == "__main__":
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default="training.csv", help="input training data file name")
    parser.add_argument("--testing", default="testing.csv", help="input testing data file name")
    parser.add_argument("--output", default="output.csv", help="output file name")
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.

    # settings
    WINDOW_SIZE = 15
    REF_DAY = 30
    PREDICT_DAY = 1
    RATIO = 0.8
    PATIENCE = 20 # 容忍度
    BATCH_SIZE = 32
    EPOCHS = 30
    
    # load training data
    df1 = pd.read_csv(args.training, header=None, usecols=[0, 1, 2, 3], names=["open", "high", "low", "close"])
    
    # display training data
    df_train = df1
    fig, ax = plt.subplots(2, figsize=(20, 10))
    ax[0].plot(df_train['open'], label='raw')
    ax[0].set_title('Raw Data')
    ax[0].set_xlabel('index')
    ax[0].set_ylabel('open price')
    ax[0].legend()
    
    # moving average(MA) for "open"
    df_train_MA = df1
    df_train_MA["open"] = df_train_MA["open"].rolling(WINDOW_SIZE).mean()
    # padding begin for data after MA
    for i in range(WINDOW_SIZE):
        df_train_MA["open"][i] = df_train_MA["open"][WINDOW_SIZE - 1]
    
    # display MA data
    ax[1].plot(df_train_MA['open'], label='MA')
    ax[1].set_title('MA Data')
    ax[1].set_xlabel('index')
    ax[1].set_ylabel('open price')
    ax[1].legend()
    # plt.show()
    plt.savefig('MA.png')
    plt.close()

    MIN, MAX = np.min(df_train_MA['open'].values), np.max(df_train_MA['open'].values)
    DIFF = MAX - MIN 
    # print("max : {}, min : {}, diff : {}".format(MAX, MIN, DIFF))
    
    # normalize data
    df_train_MA_norm = norm(df_train_MA)
    
    # set XY data(training data / label data)
    x_raw, y_raw = setXY(df_train_MA_norm, REF_DAY, PREDICT_DAY)
    
    # shuffle data
    x_train, y_train = shuffle(x_raw[:int(x_raw.shape[0] * RATIO)], y_raw[:int(y_raw.shape[0] * RATIO)])
    x_test , y_test  = shuffle(x_raw[int(x_raw.shape[0] * RATIO):], y_raw[int(y_raw.shape[0] * RATIO):])
    
    # training model setup
    lstm_model = buildLSTM(x_train.shape, PREDICT_DAY)
    early_stopping = EarlyStopping(monitor='val_mse', patience=PATIENCE, verbose=1, mode='min')
    history = lstm_model.fit(x_train, y_train, verbose=1, callbacks=[early_stopping],
        validation_data=(x_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS)

    fig, ax = plt.subplots()
    ax.plot(history['loss'], label='loss')
    ax.plot(history['mse'], label='mse')
    ax.set_ylabel('result')
    ax.set_xlabel('epoch')
    ax.set_title('history')
    ax.legend()
    plt.savefig('loss.png')
    plt.close()
    
    # # save model
    # model_name = '{}_{}.h5'.format(int(time.time()), np.around(np.min(history.history['val_mse']), decimals=4))
    # lstm_model.save(model_name)
    
    # load testing data
    df2 = pd.read_csv(args.testing, header=None, usecols=[0,1,2,3], names=['open', 'high', 'low', 'close'])
    
    # normalize testing data by min/diff value of training data
    df_test = df2.apply(lambda x: (x - MIN) / DIFF)

    # start predict
    predict_input_raw = np.array(df_train_MA_norm.iloc[-REF_DAY:]) # training data 後30天
    predict_input = np.array([predict_input_raw])
    
    predict_output = reNorm(lstm_model.predict(predict_input)[0], MIN, DIFF) # 第31天預測結果

    # record predict result
    # predict_res = [reNorm(df_train_MA_norm.iloc[-1:]['open'], MIN, DIFF).values[0]]
    predict_res = []
    predict_res.append(predict_output[0])

    for i in range(df_test.shape[0] - 1): # 做剩下的19次
        predict_input_raw = np.vstack((predict_input_raw, df_test.iloc[i]))[1:] # 將新的一天的結果貼到原始資料後面 並取後30天(過濾掉第1天)
        predict_input = np.array([predict_input_raw])
        
        predict_output = reNorm(lstm_model.predict(predict_input)[0], MIN, DIFF)
        
        predict_res.append(predict_output[0])

    print('predict price result: ', predict_res)
    
    # make decision
    hold = 0
    output_res = []
    for i in range(len(predict_res) - 1):
        assert hold >= -1 or hold <= 1, 'hold value wrong...'
        action = 0
        if predict_res[i + 1] > predict_res[i]: # tomorrow up
            if hold == 1:
                action = 0
            elif hold == 0:
                action = 1
            elif hold == -1:
                action = 1
        elif predict_res[i + 1] < predict_res[i]: # tomorrow down
            if hold == 1:
                action = -1
            elif hold == 0:
                action = -1
            elif hold == -1:
                action = 0
                
        hold += action
        output_res.append(action)
    print('predict action result: ',output_res)
    
    # write output
    with open(args.output, "w") as output_file:
        for action in output_res:
            output_file.write('{}\n'.format(action))
            
    print('Done.')
    