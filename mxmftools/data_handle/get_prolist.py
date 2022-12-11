def get_prolist(atoms, orbitals, symbols, orbitalnum):
    print(atoms, orbitals)
    atomsgroup = []
    orbitalsgroup = []
    group = []
    if atoms == ():
        atoms = ["all"]
    if orbitals == ():
        orbitals = ["all"]

    for atom in atoms:
        atomstr = "".join(atom)
        atomslist = atomstr.split(",")
        group.append(len(atomslist))
        for i in atomslist:
            if i == "all":
                atomsgroup.append(list(range(len(symbols))))
            else:
                atomsgroup.append(i.split())
    for orbital in orbitals:
        orbitalstr = "".join(orbital)
        orbitalslist = orbitalstr.split(",")
        print(orbitalslist)
        for i in orbitalslist:
            if i == "all":
                orbitalsgroup.append(list(range(orbitalnum)))
            else:
                orbitalsgroup.append(i.split())

    total_orbitals = [
        "s",
        "py",
        "pz",
        "px",
        "dxy",
        "dyz",
        "dz2",
        "dxz",
        "dx2-y2",
        "fy3x2",
        "fxyz",
        "fyz2",
        "fz3",
        "fxz2",
        "fzx2",
        "fx3",
    ]
    prolist = []
    for i, proatoms in enumerate(atomsgroup):
        atomnumlist = []
        for atom in proatoms:
            try:
                atomnumlist.append(int(atom))
            except:
                for index, symbol in enumerate(symbols):
                    if symbol == atom:
                        atomnumlist.append(index)
        orbitalnumlist = []
        for orbital in orbitalsgroup[i]:
            try:
                orbitalnumlist.append(int(orbital))
            except:
                for index, orbitals in enumerate(total_orbitals):
                    if orbital in orbitals:
                        orbitalnumlist.append(index)
        prolist.append([atomnumlist, orbitalnumlist])
        print(
            f"the projected atoms are {[symbols[i] + ' ' + str(i) for i in atomnumlist]} and projected orbitals are {[total_orbitals[i] for i in orbitalnumlist]}"
        )
    proiter = (i for i in prolist)
    result = [[next(proiter) for i in range(g)] for g in group]
    return result
