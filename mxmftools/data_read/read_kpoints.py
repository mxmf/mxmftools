# -*- coding: utf-8 -*-
import re


class ReadKpoints(object):
    def __init__(self, file="KPOINTS"):
        self.file = file
        self.symbols = []
        self.division = None
        self.k_coors = []
        self.lines = self.readlines()

    def readlines(self):
        with open(self.file) as fk:
            line_num = 0
            while line_num < 4:
                line = fk.readline()
                if line.split() == []:
                    ...
                else:
                    line_num = line_num + 1
                    if line_num == 2:
                        self.division = int(line)
            while True:
                line = fk.readline()
                if not line:
                    break
                if line.split() == []:
                    ...
                else:
                    k_coor = [float(i) for i in line.split()[:3]]
                    if self.k_coors == [] or k_coor != self.k_coors[-1]:
                        self.k_coors.append(k_coor)
                        if len(line.split()) == 4:
                            symbol = line.split()[3]
                            if symbol not in [chr(i) for i in range(65, 91)] + [
                                chr(i) for i in range(97, 123)
                            ]:
                                self.symbols.append(f"${symbol}$")
                            else:
                                self.symbols.append(symbol)


if __name__ == "__main__":
    data = ReadKpoints("KPOINTS")
    print(data.division)
    print(data.k_coors)
    print(data.symbols)
