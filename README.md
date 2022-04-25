# DSAI-HW2-2021 Auto Traing

## 資料前處理
* 使用moving average對open欄位進行smoothing。
* window_size設定為15天/3週。
* MA後，前方空白Nan值用第一筆open資料填入。
```python
df_train_MA["open"] = df_train_MA["open"].rolling(WINDOW_SIZE).mean()
# padding begin for data after MA
for i in range(WINDOW_SIZE):
    df_train_MA["open"][i] = df_train_MA["open"][WINDOW_SIZE - 1]
```
![image](https://github.com/qq21629489/AutoTrading/blob/main/MA.png)

## LSTM model
* 共由6層lstm建立。![image](https://github.com/qq21629489/AutoTrading/blob/main/lstm_model.png)
* 參數設定
```python
REF_DAY = 30     # 輸入天數
PREDICT_DAY = 1  # 預測天數
RATIO = 0.8      # 訓練：測試 = 0.8 : 0.2
PATIENCE = 20    # 容忍度，超過20次loss沒有改善即early stopping
BATCH_SIZE = 32  
EPOCHS = 200     
```
* loss下降狀況。![image](https://github.com/qq21629489/AutoTrading/blob/main/loss.png)

## 預測
* 使用training.csv中後30天資料作為輸入丟進model，預測第一天open值。
* 接下來將testing.csv中第一天資料放到30天資料的尾巴，並移除第一筆資料，並丟進model預測第二天open值。
* 重複上一步直到testing.csv沒有資料為止。

## 動作選擇
* 若明天漲
    * 現在有1股票 ：不動 action = 0
    * 現在有0股票 ：買入 action = 1
    * 現在有-1股票：買入 action = 1
* 若明天跌
    * 現在有1股票 ：賣出 action = -1
    * 現在有0股票 ：賣出 action = -1
    * 現在有-1股票：不動 action = 0

## args
* training data: training.csv
* testing data : testing.csv
* output file  : output.csv