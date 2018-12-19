#!/usr/bin/python
# -*- coding: UTF-8 -*-


import numpy as np


from config.UDF import output_dataset


def data_split(model_type):
    # load dataset [1780, 192, 192, 1]
    datapath = ".//dataset//npy//"
    list_number = np.asarray(range(20))
    list_number = np.delete(list_number, LOOCV)
    #     np.random.shuffle(list_number)
    list_train = []
    list_val = []
    for i in range(5):
        list_val.append(list_patient[list_number[i]])
    for i in range(14):
        list_train.append(list_patient[list_number[i + 4]])
    output_dataset(log_path + model_type + '_' + str(LOOCV), list_train, list_val, LOOCV)

    x_test = np.zeros((89 * 1, 192, 192, slice_x), dtype=np.float32)
    y_test = np.zeros((89 * 1, 192, 192, 1), dtype=np.float32)
    x_val = np.zeros((89 * 5, 192, 192, slice_x), dtype=np.float32)
    y_val = np.zeros((89 * 5, 192, 192, 1), dtype=np.float32)
    x_train = np.zeros((89 * 14, 192, 192, slice_x), dtype=np.float32)
    y_train = np.zeros((89 * 14, 192, 192, 1), dtype=np.float32)

    # MR-only model
    if model_type == 'MR-only':
        amp_x = 3000
        amp_y = 14000
        temp_x = np.load(datapath + list_patient[LOOCV] + '_water_data.npy')
        temp_y = np.load(datapath + list_patient[LOOCV] + '_5min_data.npy')
        for idx in range(89):
            x_test[idx, :, :, :] = temp_x[:, :, idx].reshape((1, 192, 192, slice_x)) / amp_x
            y_test[idx, :, :, :] = temp_y[:, :, idx].reshape((1, 192, 192, 1)) / amp_y

        idx_x = 0
        idx_y = 0
        for patient in list_val:
            name_x = datapath + patient + '_water_data.npy'
            name_y = datapath + patient + '_5min_data.npy'
            temp_x = np.load(name_x)
            temp_y = np.load(name_y)
            for idx in range(89):
                x_val[idx_x, :, :, :] = temp_x[:, :, idx].reshape((1, 192, 192, slice_x)) / amp_x
                y_val[idx_y, :, :, :] = temp_y[:, :, idx].reshape((1, 192, 192, 1)) / amp_y
                idx_x = idx_x + 1
                idx_y = idx_y + 1
        x_val = x_val
        y_val = y_val

        idx_x = 0
        idx_y = 0
        for patient in list_train:
            name_x = datapath + patient + '_water_data.npy'
            name_y = datapath + patient + '_5min_data.npy'
            temp_x = np.load(name_x)
            temp_y = np.load(name_y)
            for idx in range(89):
                x_train[idx_x, :, :, :] = temp_x[:, :, idx].reshape((1, 192, 192, slice_x)) / amp_x
                y_train[idx_y, :, :, :] = temp_y[:, :, idx].reshape((1, 192, 192, 1)) / amp_y
                idx_x = idx_x + 1
                idx_y = idx_y + 1

                # PET-only model
    if model_type == 'PET-only':
        amp_x = 14000
        amp_y = 14000
        temp_x = np.load(datapath + list_patient[LOOCV] + '_1min_data.npy')
        temp_y = np.load(datapath + list_patient[LOOCV] + '_5min_data.npy')
        for idx in range(89):
            if idx == 0:
                idx_0 = 0
                idx_1 = 0
                idx_2 = 1
            if idx == 88:
                idx_0 = 87
                idx_1 = 88
                idx_2 = 88
            if idx > 0 and idx < 88:
                idx_0 = idx - 1
                idx_1 = idx
                idx_2 = idx + 1
            x_0 = temp_x[:, :, idx_0].reshape((192, 192, 1))
            x_1 = temp_x[:, :, idx_1].reshape((192, 192, 1))
            x_2 = temp_x[:, :, idx_2].reshape((192, 192, 1))
            x_stack = np.concatenate((x_0, x_1, x_2), axis=2)
            x_test[idx, :, :, :] = x_stack.reshape((1, 192, 192, slice_x)) / amp_x
            y_test[idx, :, :, :] = temp_y[:, :, idx].reshape((1, 192, 192, 1)) / amp_y

        idx_x = 0
        idx_y = 0
        for patient in list_val:
            name_x = datapath + patient + '_1min_data.npy'
            name_y = datapath + patient + '_5min_data.npy'
            temp_x = np.load(name_x)
            temp_y = np.load(name_y)
            for idx in range(89):
                if idx == 0:
                    idx_0 = 0
                    idx_1 = 0
                    idx_2 = 1
                if idx == 88:
                    idx_0 = 87
                    idx_1 = 88
                    idx_2 = 88
                if idx > 0 and idx < 88:
                    idx_0 = idx - 1
                    idx_1 = idx
                    idx_2 = idx + 1
                x_0 = temp_x[:, :, idx_0].reshape((192, 192, 1))
                x_1 = temp_x[:, :, idx_1].reshape((192, 192, 1))
                x_2 = temp_x[:, :, idx_2].reshape((192, 192, 1))
                x_stack = np.concatenate((x_0, x_1, x_2), axis=2)
                x_val[idx_x, :, :, :] = x_stack.reshape((1, 192, 192, slice_x)) / amp_x
                y_val[idx_y, :, :, :] = temp_y[:, :, idx].reshape((1, 192, 192, 1)) / amp_y
                idx_x = idx_x + 1
                idx_y = idx_y + 1

        idx_x = 0
        idx_y = 0
        for patient in list_train:
            name_x = datapath + patient + '_1min_data.npy'
            name_y = datapath + patient + '_5min_data.npy'
            temp_x = np.load(name_x)
            temp_y = np.load(name_y)
            for idx in range(89):
                if idx == 0:
                    idx_0 = 0
                    idx_1 = 0
                    idx_2 = 1
                if idx == 88:
                    idx_0 = 87
                    idx_1 = 88
                    idx_2 = 88
                if idx > 0 and idx < 88:
                    idx_0 = idx - 1
                    idx_1 = idx
                    idx_2 = idx + 1
                x_0 = temp_x[:, :, idx_0].reshape((192, 192, 1))
                x_1 = temp_x[:, :, idx_1].reshape((192, 192, 1))
                x_2 = temp_x[:, :, idx_2].reshape((192, 192, 1))
                x_stack = np.concatenate((x_0, x_1, x_2), axis=2)
                x_train[idx_x, :, :, :] = x_stack.reshape((1, 192, 192, slice_x)) / amp_x
                y_train[idx_y, :, :, :] = temp_y[:, :, idx].reshape((1, 192, 192, 1)) / amp_y
                idx_x = idx_x + 1
                idx_y = idx_y + 1

                # PET-MR model
    if model_type == 'PET-MR':
        amp_x_PET = 14000
        amp_x_MR = 3000
        amp_y = 14000
        temp_x_PET = np.load(datapath + list_patient[LOOCV] + '_1min_data.npy')
        temp_x_MR = np.load(datapath + list_patient[LOOCV] + '_water_data.npy')
        temp_y = np.load(datapath + list_patient[LOOCV] + '_5min_data.npy')
        for idx in range(89):
            if idx == 0:
                idx_0 = 0
                idx_1 = 0
                idx_2 = 1
            if idx == 88:
                idx_0 = 87
                idx_1 = 88
                idx_2 = 88
            if idx > 0 and idx < 88:
                idx_0 = idx - 1
                idx_1 = idx
                idx_2 = idx + 1
            x_0 = temp_x_PET[:, :, idx_0].reshape((192, 192, 1)) / amp_x_PET
            x_1 = temp_x_PET[:, :, idx_1].reshape((192, 192, 1)) / amp_x_PET
            x_2 = temp_x_PET[:, :, idx_2].reshape((192, 192, 1)) / amp_x_PET
            x_3 = temp_x_MR[:, :, idx_1].reshape((192, 192, 1)) / amp_x_MR
            x_stack = np.concatenate((x_0, x_1, x_2, x_3), axis=2)
            x_test[idx, :, :, :] = x_stack.reshape((1, 192, 192, slice_x))
            y_test[idx, :, :, :] = temp_y[:, :, idx].reshape((1, 192, 192, 1)) / amp_y

        idx_x = 0
        idx_y = 0
        for patient in list_val:
            name_x_PET = datapath + patient + '_1min_data.npy'
            name_x_MR = datapath + patient + '_water_data.npy'
            name_y = datapath + patient + '_5min_data.npy'
            temp_x_PET = np.load(name_x_PET)
            temp_x_MR = np.load(name_x_MR)
            temp_y = np.load(name_y)
            for idx in range(89):
                if idx == 0:
                    idx_0 = 0
                    idx_1 = 0
                    idx_2 = 1
                if idx == 88:
                    idx_0 = 87
                    idx_1 = 88
                    idx_2 = 88
                if idx > 0 and idx < 88:
                    idx_0 = idx - 1
                    idx_1 = idx
                    idx_2 = idx + 1
                x_0 = temp_x_PET[:, :, idx_0].reshape((192, 192, 1)) / amp_x_PET
                x_1 = temp_x_PET[:, :, idx_1].reshape((192, 192, 1)) / amp_x_PET
                x_2 = temp_x_PET[:, :, idx_2].reshape((192, 192, 1)) / amp_x_PET
                x_3 = temp_x_MR[:, :, idx_1].reshape((192, 192, 1)) / amp_x_MR
                x_stack = np.concatenate((x_0, x_1, x_2, x_3), axis=2)
                x_val[idx_x, :, :, :] = x_stack.reshape((1, 192, 192, slice_x))
                y_val[idx_y, :, :, :] = temp_y[:, :, idx].reshape((1, 192, 192, 1)) / amp_y
                idx_x = idx_x + 1
                idx_y = idx_y + 1

        idx_x = 0
        idx_y = 0
        for patient in list_train:
            name_x_PET = datapath + patient + '_1min_data.npy'
            name_x_MR = datapath + patient + '_water_data.npy'
            name_y = datapath + patient + '_5min_data.npy'
            temp_x_PET = np.load(name_x_PET)
            temp_x_MR = np.load(name_x_MR)
            temp_y = np.load(name_y)
            for idx in range(89):
                if idx == 0:
                    idx_0 = 0
                    idx_1 = 0
                    idx_2 = 1
                if idx == 88:
                    idx_0 = 87
                    idx_1 = 88
                    idx_2 = 88
                if idx > 0 and idx < 88:
                    idx_0 = idx - 1
                    idx_1 = idx
                    idx_2 = idx + 1
                x_0 = temp_x_PET[:, :, idx_0].reshape((192, 192, 1)) / amp_x_PET
                x_1 = temp_x_PET[:, :, idx_1].reshape((192, 192, 1)) / amp_x_PET
                x_2 = temp_x_PET[:, :, idx_2].reshape((192, 192, 1)) / amp_x_PET
                x_3 = temp_x_MR[:, :, idx_1].reshape((192, 192, 1)) / amp_x_MR
                x_stack = np.concatenate((x_0, x_1, x_2, x_3), axis=2)
                x_train[idx_x, :, :, :] = x_stack.reshape((1, 192, 192, slice_x))
                y_train[idx_y, :, :, :] = temp_y[:, :, idx].reshape((1, 192, 192, 1)) / amp_y
                idx_x = idx_x + 1
                idx_y = idx_y + 1