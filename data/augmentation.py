#!/usr/bin/python
# -*- coding: UTF-8 -*-


from keras.preprocessing.image import ImageDataGenerator


def data_aug(conf_a, x_train, y_train, x_val, y_val):

    # train data_generator
    data_generator1 = ImageDataGenerator(rotation_range=conf_a["rotation_range"],
                                         shear_range=conf_a["shear_range"],
                                         width_shift_range=conf_a["width_shift_range"],
                                         height_shift_range=conf_a["height_shift_range"],
                                         zoom_range=conf_a["zoom_range"],
                                         horizontal_flip=conf_a["horizontal_flip"],
                                         vertical_flip=conf_a["vertical_flip"],
                                         fill_mode=conf_a["fill_mode"])
    data_generator2 = ImageDataGenerator(rotation_range=conf_a["rotation_range"],
                                         shear_range=conf_a["shear_range"],
                                         width_shift_range=conf_a["width_shift_range"],
                                         height_shift_range=conf_a["height_shift_range"],
                                         zoom_range=conf_a["zoom_range"],
                                         horizontal_flip=conf_a["horizontal_flip"],
                                         vertical_flip=conf_a["vertical_flip"],
                                         fill_mode=conf_a["fill_mode"])

    # validation data_generator
    data_generator3 = ImageDataGenerator(width_shift_range=conf_a["width_shift_range"],
                                         height_shift_range=conf_a["height_shift_range"],
                                         zoom_range=conf_a["zoom_range"],
                                         horizontal_flip=conf_a["horizontal_flip"],
                                         vertical_flip=conf_a["vertical_flip"],
                                         fill_mode=conf_a["fill_mode"])
    data_generator4 = ImageDataGenerator(width_shift_range=conf_a["width_shift_range"],
                                         height_shift_range=conf_a["height_shift_range"],
                                         zoom_range=conf_a["zoom_range"],
                                         horizontal_flip=conf_a["horizontal_flip"],
                                         vertical_flip=conf_a["vertical_flip"],
                                         fill_mode=conf_a["fill_mode"])

    # set generator
    data_generator1.fit(x_train, seed=conf_a["seed"])
    data_generator2.fit(y_train, seed=conf_a["seed"])
    data_generator3.fit(x_val, seed=conf_a["seed"])
    data_generator4.fit(y_val, seed=conf_a["seed"])
    data_generator_t = zip(data_generator1.flow(x=x_train, y=None,
                                                batch_size=conf_a["batch_size"], seed=conf_a["seed"]),
                           data_generator2.flow(x=y_train, y=None,
                                                batch_size=conf_a["batch_size"], seed=conf_a["seed"]))
    data_generator_v = zip(data_generator3.flow(x=x_val, y=None,
                                                batch_size=conf_a["batch_size"], seed=conf_a["seed"]),
                           data_generator4.flow(x=y_val, y=None,
                                                batch_size=conf_a["batch_size"], seed=conf_a["seed"]))

    return data_generator_v, data_generator_t
