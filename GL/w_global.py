#!/usr/bin/python
# -*- coding: UTF-8 -*-


def __init__():
    global GDICT
    GDICT = {}


def GL_set_value(key, value):
    GDICT[key] = value


def GL_get_value(key, defvalue=None):
    try:
        return GDICT[key]
    except KeyError:
        return defvalue
