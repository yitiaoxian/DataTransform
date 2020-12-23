#!/usr/bin/env/ python
# -*- coding:utf-8 -*-


# @Time    :2020/10/30 16:20
# @Author  :xiaoqianke

import psycopg2 as psy
import numpy as np
import pandas as pd




PG_SQL_LOCAL = {
    'database':'backup_58',
    'user':'postgres',
    'password':'123456',
    #localhost
    'host':'localhost',
    'port':'5432'
}



def selectOpreation():
    conn_ = psy.connect(**PG_SQL_LOCAL)
    print('connect successful')
    cur_ = conn_.cursor()
    conn_.setAutoCommit(False)

    count = 0
    while True:
        cur_.execute("select * from \"T_MDC_HISTORY_DATA_copy\" limit 1; ")

        rows_ = cur_.fetchmany(size=1)
        cur_.next()
        count += 1
        print('第%d次查询' % (count)+'结果')
        if not rows_:
            print('BREAK')
            break
        for row in rows_:
            transformData()
            print(row)
    conn_.close()

def transformData():
    print('connect s2')

def transactionControl():
    connection = psy.connect(**PG_SQL_LOCAL)


selectOpreation()