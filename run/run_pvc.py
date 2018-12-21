#!/usr/bin/python
# -*- coding: UTF-8 -*-

def w_train(model, X, Y):

    curr_loss = model.train_on_batch(X, Y)


    return curr_loss