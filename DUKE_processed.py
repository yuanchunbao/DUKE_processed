import argparse
import os
from collections import namedtuple
import gc
import os.path as osp
import json
import pandas as pd
import logging
from multiprocessing import Pool, Manager
from queue import Queue
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pydicom
import torch
from torch.nn.functional import interpolate


class Series:
    def __init__(
            self,
            patient_id: str,
            category: str,
            confidence: str,
            image_path: str,
            num_slices: int,
            spacing: List[float],
            slices: List[str] = None,
            **kwargs
    ):
        self.patient_id = patient_id
        self.category = category
        self.confidence = confidence
        self.image_path = image_path
        self.image = None
        # self.hdf5_dataset_path, self.key = self.image_path.split(':')
        self.num_slices = num_slices
        self.spacing = spacing
        self.slices = slices if slices is not None else []

    @property
    def name(self) -> str:
        # mvd_index = int(self.patient_id.split('-')[1])
        return self.patient_id

    def get_image(
            self
    ) -> Optional[np.ndarray]:
        image = np.load(self.image_path)
        return image

    def get_liver_mask(self) -> Optional[np.ndarray]:
        liver_mask_path = osp.join(
            '/ssd/SeriesCls/DCE_organ_seg', self.name + '_ogn.npy')
        if osp.isfile(liver_mask_path):
            liver_mask = np.load(liver_mask_path)
            liver_mask = liver_mask == 1

            return liver_mask

    def to_dict(self) -> Dict:
        return {
            'patient_id': self.patient_id,
            'category': self.category,
            'confidence': self.confidence,
            'image_path': self.image_path,
            'num_slices': self.num_slices,
            'spacing': self.spacing,
            'slices': self.slices}


    def load_raw_dataset(self, path: str):
        with open(path) as f:
            dataset_dict = json.load(f)
            for sub_dataset_path in dataset_dict.pop(
                    'existed_data_jsons', []):
                sub_dataset_dict = json.load(open(sub_dataset_path))
                for category, series_info_list in sub_dataset_dict.items():
                    if category not in self.categories:
                        continue
                    series_list = self.dataset.setdefault(category, [])
                    for series_info in series_info_list:
                        series_info['image_path'] = series_info.pop(
                            'pixel_array')
                        series_info.setdefault('confidence', 'High')
                        series = Series(**series_info)
                        series_list.append(series)

def serialize_dataset(dataset_row) -> Dict:
    dataset = {}
    for category, series_list in dataset_row.items():
        dataset[category] = [series.to_dict() for series in series_list]
    return dataset

def save_as(dataset_row, path: str) -> None:
    json.dump(serialize_dataset(dataset_row), open(path, 'w'), indent=2)

def determine_shape_by_spacing(
        original_shape,
        original_spacing,
        targrt_spacing
):
    assert len(original_shape) >= len(original_spacing), \
    f"Spacing {original_spacing} is not consistent with shape"
    assert len(original_spacing) >= len(targrt_spacing), \
    f"Targrt spacing {targrt_spacing} is not consistent with the" \
    f"original one {original_spacing}."
    if len(original_spacing) > len(targrt_spacing):
        original_spacing = original_spacing[-len(targrt_spacing):]
    target_shape = []
    for i in range(-1, -len(original_spacing) - 1, -1):
        _original_shape = original_shape[i]
        _original_spacing = original_spacing[i]
        _target_spacing = targrt_spacing[i]
        _new_shape = np.round(
            (_original_shape - 1) * _original_spacing / _target_spacing) +1
        _new_shape = int(_new_shape)
        target_shape.insert(0, _new_shape)
    target_shape = list(
        original_shape[:(len(original_shape) - len(original_spacing))]) \
        + target_shape
    return tuple(target_shape)

def change_spacing(
        image,
        *,
        original_spacing,
        target_spacing,
        **kwargs_for_interpolation
):
    assert 2< len(image.shape) < 5
    target_shape = determine_shape_by_spacing(
        original_shape=image.shape,
        original_spacing=original_spacing,
        targrt_spacing=target_spacing
    )
    _image = image.to(dtype=torch.float32)
    original_number_of_dimensions = len(_image.shape)
    while len(_image.shape) < 5:
        _image.unsqueeze_(dim=0)
    _image = interpolate(_image, size=target_shape, **kwargs_for_interpolation)
    while len(_image.shape) > original_number_of_dimensions:
        _image.squeeze_(dim=0)
    return _image

def record_spacing(folder_path, output_path, SeriesClassificationKey, SequenceTypes):
    SequenceTypes_dict = {}
    for index, item in SequenceTypes.iterrows():
        SequenceTypes_dict[item[0]] = item[2]

    dataset_train_json: Dict[str, List[Series]] = {}
    dataset_val_json: Dict[str, List[Series]] = {}
    for index, item in SeriesClassificationKey.iterrows():
        patient_id = '0' * (4-len(str(item[0]))) + str(item[0]) + "_" + str(item[1])
        category = SequenceTypes_dict[item[2]]
        image_path = osp.join(output_path, '0' * (4-len(str(item[0]))) + str(item[0]), '0' * (4-len(str(item[0]))) + str(item[0]) + '_' + str(item[1]) + '.npy')
        num_slice = 0
        spacing = None
        slice_path = osp.join(folder_path, '0' * (4-len(str(item[0]))) + str(item[0]), str(item[1]))
        slice_list = []
        for slice_item in os.listdir(slice_path):
            slice_list.append(pydicom.read_file(osp.join(slice_path, slice_item)))
        slice_list.sort(key=lambda x: x.ImagePositionPatient[2])
        if hasattr(slice_list[0], 'SpacingBetweenSlices'):
            print(patient_id, 'SpacingBetweenSlices exist!', slice_list[0].PixelSpacing, slice_list[0].SpacingBetweenSlices)
            num_slice = len(slice_list)
            spacing = [slice_list[0].PixelSpacing[0], slice_list[0].PixelSpacing[1], slice_list[0].SpacingBetweenSlices]
        else:
            print(patient_id, 'SpacingBetweenSlices not exist!', slice_list[0].PixelSpacing)
            if category not in ['MRCP', 'T2-Cor', 'LOCAL', 'DCE-EP-Cor']:
                num_slice = len(slice_list)
                spacing=0.
                for index, slice_list_item in enumerate(slice_list):
                    if index != 0:
                        spacing += slice_list[index].ImagePositionPatient[2] - slice_list[index - 1].ImagePositionPatient[2]
                spacing = spacing / (len(slice_list)-1)
                print('Calculate spacing:', spacing)
                spacing = [slice_list[0].PixelSpacing[0], slice_list[0].PixelSpacing[1], spacing]
        series_info = Series(
            patient_id=patient_id,
            category=category,
            confidence='High',
            image_path=image_path,
            num_slices=num_slice,
            spacing=spacing,
            slice=None
        )

        for index, item in enumerate(slice_list):
            slice_list[index] = item.pixel_array.astype(np.float32)
        image = torch.tensor(np.array(slice_list))
        if series_info.category not in ['MRCP', 'T2-Cor', 'LOCAL', 'DCE-EP-Cor']:
            _image = change_spacing(
                image,
                original_spacing=tuple(series_info.spacing),
                target_spacing=(5, 1, 1),
                mode='trilinear',
                align_corners=True
            )
            image = _image.numpy()
        else:
            image = image.numpy()
        output_case_path = osp.join(output_path, series_info.patient_id.split('_')[0])
        if not osp.exists(output_case_path):
            os.mkdir(output_case_path)
        np.save(osp.join(output_case_path, series_info.patient_id + '.npy'), image)
        if int(series_info.patient_id.split('_')[0]) % 4 ==0:
            series_list = dataset_val_json.setdefault(category, [])
            print(' in val')
        else:
            series_list = dataset_train_json.setdefault(category, [])
        series_list.append(series_info)
    save_as(dataset_train_json, r'D:\袁春宝\DUKE_Liver_Dataset\DUKE_spacing_processed\dataset_train.json')
    save_as(dataset_val_json, r'D:\袁春宝\DUKE_Liver_Dataset\DUKE_spacing_processed\dataset_val.json')

folder_path = 'D:\袁春宝\DUKE_Liver_Dataset\Series_Classification\Series_Classification'
output_path = 'D:\袁春宝\DUKE_Liver_Dataset\DUKE_spacing_processed'
SeriesClassificationKey = pd.read_csv('D:\袁春宝\DUKE_Liver_Dataset\SeriesClassificationKey.csv')
SequenceTypes = pd.read_csv('D:\袁春宝\DUKE_Liver_Dataset\SequenceTypes.csv')
record_spacing(folder_path, output_path, SeriesClassificationKey, SequenceTypes)