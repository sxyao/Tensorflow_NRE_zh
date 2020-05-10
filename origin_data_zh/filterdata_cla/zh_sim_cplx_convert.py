# coding: utf-8
from langconv import *


def convert_cplx2sim(entity):
    # 转换繁体到简体
    entity = Converter('zh-hans').convert(entity.decode('utf-8'))
    entity = entity.encode('utf-8')
    return entity


def convert_sim2cplx(entity):
    # 转换简体到繁体
    entity = Converter('zh-hant').convert(entity.decode('utf-8'))
    entity = entity.encode('utf-8')
    return entity

