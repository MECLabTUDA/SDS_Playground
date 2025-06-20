import numpy as np
import torch
from PIL import ImageColor

CLASSES = {
    0: {
        "name": "Background",
        "color": '#000000'
    },
    1: {
        "name": "Cornea",
        "color": '#AF3235'
    },
    2: {
        "name": "Pupil",
        "color": '#DFE674'
    },
    3: {
        "name": "Lens",
        "color": '#AF72B0'
    },
    4: {
        "name": "Slit Knife",
        "color": '#46C5DD'
    },
    5: {
        "name": "Gauge",
        "color": '#F282B4'
    },
    6: {
        "name": "Capsulorhexis Cystotome",
        "color": '#98CC70'
    },
    7: {
        "name": "Spatula",
        "color": '#671800'
    },
    8: {
        "name": "Phacoemulsification Tip",
        "color": '#009B55'
    },
    9: {
        "name": "Irrigation-Aspiration",
        "color": '#F7921D'
    },
    10: {
        "name": "Lens Injector",
        "color": '#613F99'
    },
    11: {
        "name": "Incision Knife",
        "color": '#46C5DD'
    },
    12: {
        "name": "Katena Forceps",
        "color": '#EE2967'
    },
    13: {
        "name": "Capsulorhexis Forceps",
        "color": '#0071BC'
    }
}


def get_cataract1k_colormap():
    colors = []
    for i in range(len(CLASSES)):
        rgb_color = ImageColor.getrgb(CLASSES[i]['color'])
        colors.append(rgb_color)
    return np.array(colors)


def get_cataract1k_float_cmap():
    return torch.from_numpy(get_cataract1k_colormap())/255.0
