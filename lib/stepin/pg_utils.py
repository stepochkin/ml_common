# encoding=utf-8

import logging
import os
import random
from time import sleep

import psycopg2
import psycopg2.extras
# noinspection PyUnresolvedReferences
from psycopg2.extensions import TransactionRollbackError

from stepin.db_utils import safe_cur_rollback, safe_cur_commit


class PgConFactory(object):
    def __init__(self, config):
        self.config = config

    def __call__(self):
        return psycopg2.connect(**self.config)

    @staticmethod
    def nt_cursor(con):
        return con.cursor(cursor_factory=psycopg2.extras.NamedTupleCursor)


def role_exists(cur, name):
    cur.execute("""\
SELECT EXISTS (
  SELECT 1 FROM pg_roles r WHERE r.rolname = %(name)s
)
""", {'name': name.lower()}
    )
    return cur.fetchall()[0][0]


def db_exists(cur, name):
    cur.execute("""\
SELECT EXISTS (
  SELECT 1 FROM pg_database d WHERE d.datname = %(name)s
)
""", {'name': name.lower()}
    )
    return cur.fetchall()[0][0]


def relation_exists(cur, name, schema=None):
    cur.execute("""\
SELECT EXISTS (
  SELECT 1
  FROM
    pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
  WHERE
    c.relname = %(name)s AND
    n.nspname = {schema}
)
""".format(schema='current_schema()' if schema is None else '%(schema)s'),
        {'name': name.lower(), 'schema': schema}
    )
    return cur.fetchall()[0][0]


def table_exists(cur, tname, schema=None):
    cur.execute("""\
SELECT EXISTS (
   SELECT 1
   FROM information_schema.tables 
   WHERE
     table_schema = {schema} and
     table_catalog = current_catalog and
     table_name = %(tname)s
)
""".format(schema='current_schema()' if schema is None else '%(schema)s'),
        {'tname': tname.lower(), 'schema': schema}
    )
    return cur.fetchall()[0][0]


def exec_sql_serial(cur, sql, sql_args=None):
    if sql_args is None:
        sql_args = []
    isolation_level = cur.connection.isolation_level
    cur.connection.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_SERIALIZABLE)
    try:
        attempt_count = 0
        while True:
            if attempt_count > 3:
                raise Exception('Failed to get task batch')
            try:
                cur.execute(sql, sql_args)
                items = cur.fetchall()
            except TransactionRollbackError:
                logging.info('Process %s exec sql attempt #%s error', os.getpid(), attempt_count)
                # warn_exc('get batch attempt #%s error', attempt_count)
                safe_cur_rollback(cur)
                sleep(random.random())
            except:
                safe_cur_rollback(cur)
                raise
            else:
                safe_cur_commit(cur)
                break
            attempt_count += 1
        return items
    finally:
        cur.connection.set_isolation_level(isolation_level)
