# -*- coding: utf-8 -*-
import re


class ReadKpoints(object):
    def __init__(self, file="KPOINTS"):
        self.file = file
        self.lines = self.readlines()

    def readlines(self):
        try:
            with open(self.file) as fk:
                lines = fk.readlines()
        except:
            print(
                'open the Kponts file you specified failed , try to  read the "KPOINTS" in current directory'
            )
            try:
                with open("KPOINTS") as fk:
                    lines = fk.readlines()
            except:
                print('failed to read the "KPOINTS" in current directory')
                lines = None
        return lines

    @property
    def division(self):
        try:
            return int(self.lines[1])
        except:
            return None

    @property
    def symbols(self):
        symbollist = []
        inputlist = []
        try:
            for line in self.lines:
                if r"!" in line:
                    symbol = re.findall(r" \! (.*)", line)[0]
                    if inputlist == []:
                        inputlist.append(symbol)
                    else:
                        if symbol != inputlist[-1]:
                            inputlist.append(symbol)
                    if "$" in symbol:
                        symbol = r"\rm {}".format(symbol)
                    else:
                        symbol = r"$\rm {}$".format(symbol)
                    if symbollist == []:
                        symbollist.append(symbol)
                    else:
                        if symbol != symbollist[-1]:
                            symbollist.append(symbol)
            print(
                "The high symmetry point used in the calculation is: {}".format(
                    inputlist
                )
            )
        except:
            print("The high symmetry point used in the calculation is: []")
        return symbollist


if __name__ == "__main__":
    data = ReadKpoints("KPOINTS")
