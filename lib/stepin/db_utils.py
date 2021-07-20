# encoding=utf-8

from stepin.log_utils import warn_exc, safe_close


# noinspection PyBroadException
def safe_commit(con):
    if con is None:
        return
    try:
        con.commit()
    except Exception as e:
        if not isinstance(e, KeyboardInterrupt):
            warn_exc()
        try:
            con.rollback()
        except:
            warn_exc()
        raise


# noinspection PyBroadException
def safe_rollback(con):
    if con is None:
        return
    try:
        con.rollback()
    except KeyboardInterrupt:
        raise
    except:
        warn_exc()
        raise


# noinspection PyBroadException
def safe_cur_commit(cur):
    if cur is None:
        return
    try:
        cur.connection.commit()
    except Exception as e:
        if not isinstance(e, KeyboardInterrupt):
            warn_exc()
        try:
            cur.connection.rollback()
        except:
            warn_exc()
        raise


# noinspection PyBroadException
def safe_cur_rollback(cur):
    if cur is None:
        return
    try:
        cur.connection.rollback()
    except KeyboardInterrupt:
        raise
    except:
        warn_exc()
        raise


def get_sql_value(cur, sql, args=None):
    if args is None:
        args = tuple()
    cur.execute(sql, args)
    for r in cur:
        return r[0]


class LazyConnectExecutor(object):
    def __init__(self, con_factory):
        self.con_factory = con_factory
        self.con = None
        self.cur = None
        self.exec_proc = self._checking_execute

    def _connect(self):
        if self.con is None:
            self.con = self.con_factory()
            self.cur = self.con.cursor()
        else:
            safe_close(self.con)

    def _checking_execute(self, query, params=None):
        self._connect()
        self.exec_proc = self._raw_execute
        self._raw_execute(query, params=params)

    def _raw_execute(self, query, params=None):
        try:
            if params is None:
                self.cur.execute(query)
            else:
                self.cur.execute(query, params)
        except:
            warn_exc()
            raise

    def execute(self, query, params=None):
        self.exec_proc(query, params=params)
        return self.cur
