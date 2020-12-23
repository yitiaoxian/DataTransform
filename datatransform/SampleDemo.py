#!/usr/bin/env/ python
# -*- coding:utf-8 -*-


# @Time    :2020/11/9 9:57
# @Author  :xiaoqianke

import psycopg2 as psy

PG_SQL_LOCAL = {
    'database':'backup_58',
    'user':'postgres',
    'password':'123456',
    #localhost
    'host':'localhost',
    'port':'5432'
}

def selectSample():
    conn_ = psy.connect(**PG_SQL_LOCAL)
    print('connect successful')
    cur_ = conn_.cursor()
    conn_.setAutoCommit(False)