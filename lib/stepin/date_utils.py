import logging
from datetime import timedelta


def days_iter(from_dt, to_dt, chunk_days):
    fdt = from_dt
    tdt = fdt + timedelta(days=chunk_days - 1)
    while fdt <= to_dt:
        logging.info('%s %s', fdt, tdt)
        yield fdt, tdt
        fdt += timedelta(days=chunk_days)
        tdt = fdt + timedelta(days=chunk_days - 1)
        if tdt > to_dt:
            tdt = to_dt
