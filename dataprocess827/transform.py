#!/usr/bin/env/ python
# -*- coding:utf-8 -*-


# @Time    :2021/1/11 15:19
# @Author  :yitiaoxian

import psycopg2 as psy
import tensorflow as tf
import time
import numpy as np
import sklearn.neural_network as nn
import pandas as pd

PG_SQL_LOCAL = {
    'database':'backup_58',
    'user':'postgres',
    'password':'123456',
    #localhost postgresql
    'host':'localhost',
    'port':'5432'
}

LIMIT = 2
OFFSET = 1
#连接数据库获取数据
def _main():
    conn = psy.connect(**PG_SQL_LOCAL)
    #遍历次数 可以用来控制数量
    i = 0
    while  i < 10:
        curs = conn.cursor()
        sql_cmd = 'select * from \"T_MDC_HISTORY_DATA_Transform\" limit {} offset {};'.format(LIMIT , i*LIMIT)
        dataframe = pd.read_sql(sql_cmd,conn)
        print(dataframe)
        dataTrans(dataframe)
        #curs.execute(sql_cmd)
        #rt_list = curs.fetchall()
        #dataTrans(rt_list)
        conn.commit()
        curs.close()
        i = i + 1
        time.sleep(1)


#数据整理
def dataTrans(dataframe):
    for r in dataframe:
        dataset = pd.arrays(r)

        for _r in r :
            print(_r)
        #print('r[0] :',r[0],  '    r[1] : ',r[1])
    print("开始处理获取到的数据")


#bp神经网络
def net():
    print("")

def get_compiled_model():
    model = tf.nn

_main()