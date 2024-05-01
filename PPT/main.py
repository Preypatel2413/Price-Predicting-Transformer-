import csv
import numpy as np
from datetime import datetime

def parse_json_data(json_data):
    candles = json_data["data"]["candles"]
    for i, candle in enumerate(candles):
        candles[i][0] = (int(candle[0][11:13]) - 9)*60 + int(candle[0][14:16])
    candles.reverse()
    return np.array(candles, dtype=float)

def feature_extract(candles):

    feature_vectors = []

    for i in range(1, len(candles)):
        candle = candles[i]
        number = i
        time = candle[0]
        open_price = candle[1]
        high_price = candle[2]
        low_price = candle[3]
        close_price = candle[4]
        volume = candle[5]
        xz = candle[6]

        p_change_oc = (close_price - open_price)/ open_price
        p_change_hl = (high_price - low_price)/low_price

        sma_2 = ((sma_2 * 2 - candles[i-3][4] + candles[i][4])/2) if(i>=2) else (np.mean(candles[:i, 4]))
        sma_4 = ((sma_4 * 4 - candles[i-5][4] + candles[i][4])/4) if(i>=4) else (np.mean(candles[:i, 4]))
        sma_8 = ((sma_8 * 8 - candles[i-9][4] + candles[i][4])/8) if(i>=8) else (np.mean(candles[:i, 4]))
        sma_16 = ((sma_16 * 16 - candles[i-17][4] + candles[i][4])/16) if(i>=16) else (np.mean(candles[:i, 4]))
        sma_32 = ((sma_32 * 32 - candles[i-33][4] + candles[i][4])/32) if(i>=32) else (np.mean(candles[:i, 4]))
        sma_64 = ((sma_64 * 64 - candles[i-65][4] + candles[i][4])/64) if(i>=64) else (np.mean(candles[:i, 4]))
        sma_128 = ((sma_128 * 128 - candles[i-129][4] + candles[i][4])/128) if(i>=128) else (np.mean(candles[:i, 4]))

        ema_2 = (candles[i][4]*(2/(2+1)) + ema_2*(1-(2/(2+1)))) if(i>=2) else (np.mean(candles[:i, 4]))
        ema_4 = (candles[i][4]*(2/(4+1)) + ema_4*(1-(2/(4+1)))) if(i>=4) else ((np.mean(candles[:i, 4]) + ema_2)/2)
        ema_8 = (candles[i][4]*(2/(8+1)) + ema_8*(1-(2/(8+1)))) if(i>=8) else ((np.mean(candles[:i, 4])+ema_4)/2)
        ema_16 = (candles[i][4]*(2/(16+1)) + ema_16*(1-(2/(16+1)))) if(i>=16) else ((np.mean(candles[:i, 4])+ema_8)/2)
        ema_32 = (candles[i][4]*(2/(32+1)) + ema_32*(1-(2/(32+1)))) if(i>=32) else ((np.mean(candles[:i, 4])+ema_16)/2)
        macd = ema_2 - ema_32

        rsi = 100 - (100/ (1 + (max(candles[i][4]-candles[i-14][4], 0)/max(candles[i-14][4]-candles[i][4], 0)))) if(i>=14 and max(candles[i-14][4]-candles[i][4], 0)!=0) else(100)
        
        std = np.std(np.mean(candles[:i, 4]))
        up_band = sma_8 + (2*std)
        low_band = sma_8 - (2*std)

        vol_change = (candles[i][5] - candles[i-1][5])/candles[i][5] if (i!=0 and candles[i][5]!=0) else 0
        atr = max((candles[i][2]-candles[i][3]), abs(candles[i][2]-candles[i-1][4]), abs(candles[i][3]-candles[i-1][4])) if (i!=0) else 0
        di_p = ((candles[i][2]-candles[i-1][2])/atr) if (i!=0 and atr!=0) else 0
        di_n = ((candles[i-1][3]-candles[i][3])/atr) if (i!=0 and atr!=0) else 0
        mas = candles[i][4] - sma_8

        feature_vectors.append([number, time, open_price, high_price, low_price, close_price, volume, xz, p_change_oc, p_change_hl, sma_2, sma_4, sma_8, sma_16, sma_32, sma_64, sma_128, ema_2, ema_4, ema_8, ema_16, ema_32, macd, rsi, std, up_band, low_band, vol_change, atr, di_p, di_n, mas])

    return np.array(feature_vectors, dtype=float)

def normalize(data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    std_dev = np.where(std_dev == 0, 1e-5, std_dev)
    return  (data - mean) / std_dev

a = {"status":"success","data":{"candles":[["2023-07-26T15:29:00+05:30",194.3,194.3,194.3,194.3,0,0],["2023-07-26T15:28:00+05:30",194.3,194.3,194.3,194.3,0,0],["2023-07-26T15:27:00+05:30",195.05,195.05,194.3,194.3,200,2480],["2023-07-26T15:26:00+05:30",195.1,195.1,195.1,195.1,240,2480],["2023-07-26T15:25:00+05:30",195.9,195.9,195.9,195.9,0,0],["2023-07-26T15:24:00+05:30",195.9,195.9,195.9,195.9,40,2480],["2023-07-26T15:23:00+05:30",197,197,197,197,0,0]]}}
B = parse_json_data(a)
C = feature_extract(B)
D = normalize(C)



