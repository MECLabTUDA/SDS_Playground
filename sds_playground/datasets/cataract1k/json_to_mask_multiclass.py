from __future__ import print_function, absolute_import, division
import json
import numpy as np
import os
from glob import glob
from PIL import Image, ImageDraw
from collections import namedtuple
from tqdm import tqdm

# data_folder = '/home/yfrisch_locale/DATA/Cataract-1k/Segmentation_dataset/Annotations/Images-and-Supervisely-Annotations'
data_folder = '/local/scratch/Cataract-1K/Segmentation_dataset/Annotations/Images-and-Supervisely-Annotations'
folder = 'json'
folder_masks = 'mask'

case_list = os.listdir(data_folder)

Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'color',  # The color of this label
    'train_id'
])

instrument_names = ['Slit Knife', 'Gauge', 'Capsulorhexis Cystotome', 'Spatula', 'Phacoemulsification Tip',
                    'Irrigation-Aspiration', 'Lens Injector', 'Incision Knife', 'Katena Forceps',
                    'Capsulorhexis Forceps']

labels = [
    #       name                     id       color
    Label('background', 0, (0, 0, 0), 0),
    Label('Cornea', 1, (1, 1, 1), 1),
    Label('Pupil', 2, (2, 2, 2), 2),
    Label('Lens', 3, (3, 3, 3), 3),
    Label('instrument_names', 4, (4, 4, 4), 4)
]

for p in tqdm(range(len(case_list)), desc='Processing cases'):

    if not os.path.isdir(data_folder + '/' + case_list[p]):
        continue

    case_folder = data_folder + '/' + case_list[p]
    case_json_folder = data_folder + '/' + case_list[p] + '/' + 'ann'
    # jlist = os.listdir(case_json_folder)
    jlist = glob(case_json_folder + '/*.json')

    try:
        os.mkdir(case_folder + '/' + folder_masks)
    except Exception as e:
        print("Re-writing in the existing folder")
        print(e)

    for i in range(len(jlist)):
        with open(jlist[i], "r") as read_file:
            try:
                data = json.load(read_file)
            except json.decoder.JSONDecodeError as e:
                print("Error in file: ", jlist[i])
                print(e)
                exit()
            # print(data)
        name = jlist[i].split("/")[-1][:-5]
        objects = data['objects']
        image_Mix = Image.new(mode="L", size=(1024, 768), color=0)

        draw1 = ImageDraw.Draw(image_Mix)

        # First: Cornea
        for j in range(len(objects)):
            title = objects[j]['classTitle']
            exterior = objects[j]['points']['exterior']
            ext = []
            for k in range(len(exterior)):
                ext.append(tuple(exterior[k]))

            if title == 'Cornea':
                draw1.polygon(ext, fill=1)

        # Second: Pupil
        for j in range(len(objects)):
            title = objects[j]['classTitle']
            exterior = objects[j]['points']['exterior']
            ext = []
            for k in range(len(exterior)):
                ext.append(tuple(exterior[k]))

            if title == 'Pupil':
                draw1.polygon(ext, fill=2)

        # Third: Lens
        for j in range(len(objects)):
            title = objects[j]['classTitle']
            exterior = objects[j]['points']['exterior']
            ext = []
            for k in range(len(exterior)):
                ext.append(tuple(exterior[k]))

            if title == 'Lens':
                draw1.polygon(ext, fill=3)

        # Forth: Instruments
        for j in range(len(objects)):
            title = objects[j]['classTitle']
            exterior = objects[j]['points']['exterior']
            ext = []
            for k in range(len(exterior)):
                ext.append(tuple(exterior[k]))

            if title in instrument_names:
                # draw1.polygon(ext, fill=4)
                draw1.polygon(ext, fill=instrument_names.index(title) + 4)

        image_Mix.save(case_folder + '/' + folder_masks + '/' + name)
