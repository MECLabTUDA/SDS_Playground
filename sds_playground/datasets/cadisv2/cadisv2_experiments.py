EXP1 = {"LABEL": {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6],
    7: [7, 8, 9, 10, 11,
        12, 13, 14, 15, 16,
        17, 18, 19, 20, 21,
        22, 23, 24, 25, 26,
        27, 28, 29, 30, 31,
        32, 33, 34, 35]
},
    "CLASS": {
        0: "Pupil",
        1: "Surgical Tape",
        2: "Hand",
        3: "Eye Retractors",
        4: "Iris",
        5: "Skin",
        6: "Cornea",
        7: "Instrument"
    }
}

EXP2 = {"LABEL": {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6],
    7: [7, 8, 10, 27, 20, 32],
    8: [9, 22],
    9: [11, 33],
    10: [12, 28],
    11: [13, 21],
    12: [14, 24],
    13: [15, 18],
    14: [16, 23],
    15: [17],
    16: [19],
    255: [25, 26, 29, 30, 31, 34, 35],
},
    "CLASS": {
        0: "Pupil",
        1: "Surgical Tape",
        2: "Hand",
        3: "Eye Retractors",
        4: "Iris",
        5: "Skin",
        6: "Cornea",
        7: "Cannula",
        8: "Cap. Cystotome",
        9: "Tissue Forceps",
        10: "Primary Knife",
        11: "Ph. Handpiece",
        12: "Lens Injector",
        13: "I/A Handpiece",
        14: "Secondary Knife",
        15: "Micromanipulator",
        16: "Cap. Forceps",
        255: "Ignore",
    }
}

EXP3 = {"LABEL": {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6],
    7: [7],
    8: [8],
    9: [9],
    10: [10],
    11: [11],
    12: [12],
    13: [13],
    14: [14],
    15: [15],
    16: [16],
    17: [17],
    18: [18],
    19: [19],
    20: [20],
    21: [21],
    22: [22],
    23: [23],
    24: [24],
    255: [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
},
    "CLASS": {
        0: "Pupil",
        1: "Surgical Tape",
        2: "Hand",
        3: "Eye Retractors",
        4: "Iris",
        5: "Skin",
        6: "Cornea",
        7: "Hydro. Cannula",
        8: "Visc. Cannula",
        9: "Cap. Cystotome",
        10: "Rycroft Cannula",
        11: "Bonn Forceps",
        12: "Primary Knife",
        13: "Ph. Handpiece",
        14: "Lens Injector",
        15: "I/A Handpiece",
        16: "Secondary Knife",
        17: "Micromanipulator",
        18: "I/A Handpiece Handle",
        19: "Cap. Forceps",
        20: "R. Cannula Handle",
        21: "Ph. Handpiece Handle",
        22: "Cap. Cystotome Handle",
        23: "Sec. Knife Handle",
        24: "Lens Injector Handle",
        255: "Ignore",
    }
}


def remap_label(label: int, exp: int) -> (int, str):

    """ Returns remapped label id and label name given label and exp id."""

    if exp == 1:
        _exp = EXP1
    elif exp == 2:
        _exp = EXP2
    elif exp == 3:
        _exp = EXP3
    else:
        raise ValueError

    for k, v in _exp["LABEL"].items():

        if label in v:
            return k, _exp["CLASS"][k]

    raise ValueError("Could not remap label.")


if __name__ == "__main__":

    assert remap_label(label=24, exp=2) == (12, 'Lens Injector')
