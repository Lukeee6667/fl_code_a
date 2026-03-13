from __future__ import print_function

from datetime import datetime


def parse_no_ms(ts):
    return datetime.strptime(ts.split(",", 1)[0].strip(), "%Y-%m-%d %H:%M:%S")


def main():
    t1 = "2026-02-27 09:40:01,471"
    t2 = "2026-02-27 16:12:51,102"

    dt1 = parse_no_ms(t1)
    dt2 = parse_no_ms(t2)
    print(int((dt2 - dt1).total_seconds()))


if __name__ == "__main__":
    main()
