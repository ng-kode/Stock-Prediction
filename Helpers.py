import numpy as np

def normalize_all_windows(windows):
    normalized = []
    for window in windows:
        normalized.append([((float(p) / float(window[0])) - 1) for p in window])
    return normalized

def denormalize_all_windows(windows, windows_ori):
    denormalized = []
    for i, window in enumerate(windows):
        phead = float(windows_ori[i][0])
        denormalized.append([(phead * float(pn + 1)) for pn in window])
    return np.array(denormalized)

def load_data(filename, 
              lookback, 
              normalize_window=True, 
              shuffle_all=False,
              shuffle_train=True,
              train_ratio = 0.7):
    f = open(filename, 'r').read()
    data = f.split('\n')

    # split and re-create data in lookback form
    lookback_index = lookback + 1
    result_ori = []
    for i in range(len(data) - lookback_index):
        result_ori.append(data[i:i + lookback_index])

    # normalize windows
    if normalize_window:
        result = normalize_all_windows(result_ori)

    # convert result to numpy array for easier manipulation
    result = np.array(result)
    if normalize_window:
        result_ori = np.array(result_ori)

    # shuffle before train test split if shuffle all
    if shuffle_all:
        indices = np.arange(len(result))
        np.random.shuffle(indices)
        result = result[indices]
        if normalize_window:
            result_ori = result_ori[indices]
    
    # train test split and return arrays
    train_rows = round(train_ratio * result.shape[0])
    train = result[:int(train_rows)]
    test = result[int(train_rows):]
    if normalize_window:
        train_ori = result_ori[:int(train_rows)]
        test_ori = result_ori[int(train_rows):]
    
    # shuffle before x y split if shuffle train
    if shuffle_train and not shuffle_all:
        indices = np.arange(len(train))
        np.random.shuffle(indices)
        train = train[indices]
        if normalize_window:
            train_ori = train_ori[indices]

    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = test[:, :-1]
    y_test = test[:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    if normalize_window:
        return  (x_train, y_train), (x_test, y_test), (train_ori, test_ori)
    else:
        return  (x_train, y_train), (x_test, y_test)