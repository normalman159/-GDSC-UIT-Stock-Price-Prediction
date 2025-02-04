import numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
time_steps = 20
close_stock = np.load('Data/close_stock.npy')
close_stock = scaler.fit_transform(close_stock.reshape(-1, 1))

def predict(model, n_days : int) :
    x_input = close_stock[-time_steps:].reshape(1,-1) #1 row, many cols
    x_input_alternative = x_input[0].tolist()
    predict_value = []
    for _ in range(n_days):
            x_input = np.array(x_input_alternative[-time_steps:]).reshape(1,time_steps,1)
            y_pred = model.predict(x_input, verbose=0)
            predict_value.append(y_pred[0])
            x_input_alternative.append(y_pred[0][0])
    predict_value = scaler.inverse_transform(np.array(predict_value).reshape(-1,1))
    return predict_value
