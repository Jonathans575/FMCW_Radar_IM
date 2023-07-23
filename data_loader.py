import h5py
import numpy as np



def load_data():

    # load training data
    print("[INFO] Loading Training Data ...")
    from_filename = 'FMCW_data_t_domain_car.hdf5'
    f = h5py.File(from_filename, 'r')
    train_X_real = f['X_real'] # radar data after split
    train_X_imag = f['X_imag']
    train_Y_real = f['Y_real']
    train_Y_imag = f['Y_imag']

    # distribute the real and imaginary part into two channels
    train_X = np.zeros(shape=(np.shape(train_X_real)[0], np.shape(train_X_real)[1], np.shape(train_X_real)[2], 2))
    train_X[:, :, :, 0] = train_X_real[:, :, :]
    train_X[:, :, :, 1] = train_X_imag[:, :, :]
    train_Y = np.zeros(shape=(np.shape(train_Y_real)[0], np.shape(train_Y_real)[1], np.shape(train_Y_real)[2], 2))
    train_Y[:, :, :, 0] = train_Y_real[:, :, :]
    train_Y[:, :, :, 1] = train_Y_imag[:, :, :]

    # clear
    f.close()

    # data normalization
    print("[INFO] Data Normalization ...")

    average = 0
    for i in range(np.shape(train_X)[0]):
        temp_X = np.sqrt(pow(train_X[i, :, :, 0], 2) + pow(train_X[i, :, :, 1], 2))
        temp_Y = np.sqrt(pow(train_Y[i, :, :, 0], 2) + pow(train_Y[i, :, :, 1], 2))
        max_X = np.max(temp_X)
        max_Y = np.max(temp_Y)
        max_complex = max(max_X, max_Y)
        train_X[i] = (train_X[i] / max_complex)
        train_Y[i] = (train_Y[i] / max_complex)
        average += max_complex
    print(average / np.shape(train_X)[0])

    # data shuffle
    print("[INFO] Data shuffle ...")
    index = [i for i in range(len(train_X))]
    np.random.seed(2020)
    np.random.shuffle(index)
    train_X = train_X[index, ...]
    train_Y = train_Y[index, ...]

    return train_X, train_Y

'''
import h5py

num_SNR = 9
num_SINR = 12
num_class_samples = 20
count = 1

f = h5py.File('F://FMCW_radar_IM/FMCW_data/dataset_wjp/FMCW_data_t_domain_320points_test.hdf5', 'r')
test_X_real = f['interfered_real']
test_X_imag = f['interfered_imag']

for p in range(1,num_SINR+1):
    a = max(p-3,1)
    for i in range(a,num_SNR+1):
        print('正在处理...SINR:', (-40+5*(p-1)), 'dB ', (-35+5*(p-1)), 'dB,SNR:', (-20+5*(i-1)), 'dB')
        starting = 0
        temp = 4
        for hh in range(2,i+1):
            starting = starting + temp
            temp = temp + 1
        starting = starting * num_class_samples
        for j in range(num_class_samples*(p-1)+1 + starting, num_class_samples*p + starting):
            signal  = test_X_real[j, :, :] + i * test_X_imag[j, :, :]
            count = count + 1
            if(count == 700):
                print("-----------")
'''
