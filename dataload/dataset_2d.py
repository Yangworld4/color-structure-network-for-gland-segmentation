import os
import csv
import random
from copy import deepcopy
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

from dataload.stain_transform import StainRGBTransform
from utils.hed_transform import RGBToHEDTransform


def _read_manifest(manifest_path, sup):
    """Read image/mask paths relative to the manifest file."""
    manifest_path = os.path.abspath(manifest_path)
    manifest_dir = os.path.dirname(manifest_path)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError('Manifest not found: {}'.format(manifest_path))

    image_paths = []
    mask_paths = []
    with open(manifest_path, newline='', encoding='utf-8-sig') as manifest_file:
        reader = csv.DictReader(manifest_file)
        if not reader.fieldnames or 'image' not in reader.fieldnames:
            raise ValueError("Manifest must contain an 'image' column: {}".format(manifest_path))
        if sup and 'mask' not in reader.fieldnames:
            raise ValueError("Supervised manifest must contain a 'mask' column: {}".format(manifest_path))

        for row_number, row in enumerate(reader, start=2):
            image_value = (row.get('image') or '').strip()
            mask_value = (row.get('mask') or '').strip()
            if not image_value:
                raise ValueError('Empty image path at {}:{}'.format(manifest_path, row_number))
            if sup and not mask_value:
                raise ValueError('Empty mask path at {}:{}'.format(manifest_path, row_number))

            image_paths.append(os.path.normpath(os.path.join(manifest_dir, image_value)))
            if sup:
                mask_paths.append(os.path.normpath(os.path.join(manifest_dir, mask_value)))

    if not image_paths:
        raise ValueError('Manifest contains no samples: {}'.format(manifest_path))

    missing = [path for path in image_paths + mask_paths if not os.path.isfile(path)]
    if missing:
        preview = ', '.join(missing[:3])
        raise FileNotFoundError(
            '{} referenced file(s) do not exist (first entries: {})'.format(len(missing), preview)
        )
    return image_paths, mask_paths

class dataset_itn(Dataset):
    def __init__(self, data_dir, input1, augmentation_1, normalize_1, sup=True, num_images=None,
                 manifest_path=None, input_mode='rgb_rgb', training=True, **kwargs):
        super(dataset_itn, self).__init__()

        if manifest_path is not None:
            img_paths_1, mask_paths = _read_manifest(manifest_path, sup)
        else:
            img_paths_1 = []
            mask_paths = []
            image_dir_1 = os.path.join(data_dir, input1)
            if not os.path.isdir(image_dir_1):
                raise FileNotFoundError('Image directory not found: {}'.format(image_dir_1))
            mask_dir = os.path.join(data_dir, 'mask') if sup else None

            for image in sorted(os.listdir(image_dir_1)):
                image_path_1 = os.path.join(image_dir_1, image)
                if not os.path.isfile(image_path_1):
                    continue
                img_paths_1.append(image_path_1)
                if sup:
                    mask_paths.append(os.path.join(mask_dir, image))

        if sup:
            assert len(img_paths_1) == len(mask_paths)

        if num_images is not None:
            len_img_paths = len(img_paths_1)
            quotient = num_images // len_img_paths
            remainder = num_images % len_img_paths

            if num_images <= len_img_paths:
                img_paths_1 = img_paths_1[:num_images]
            else:
                rand_indices = torch.randperm(len_img_paths).tolist()
                new_indices = rand_indices[:remainder]

                img_paths_1 = img_paths_1 * quotient
                img_paths_1 += [img_paths_1[i] for i in new_indices]

                if sup:
                    mask_paths = mask_paths * quotient
                    mask_paths += [mask_paths[i] for i in new_indices]

        self.img_paths_1 = img_paths_1
        self.mask_paths = mask_paths
        self.augmentation_1 = augmentation_1
        self.normalize_1 = normalize_1
        self.sup = sup
        self.input_mode = input_mode
        self.stain_transform = StainRGBTransform(training=training)
        self.hed_transform = RGBToHEDTransform()
        self.training = training
        self.ucmt_appearance = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.25,
            ),
            transforms.RandomPosterize(bits=5, p=0.2),
            transforms.RandomAutocontrast(p=0.2),
            transforms.RandomEqualize(p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])
        self.kwargs = kwargs

    def _shared_geometric_transform(self, image, mask=None):
        image = Image.fromarray(image)
        mask = Image.fromarray(mask) if mask is not None else None

        if self.training:
            if random.random() < 0.5:
                image = transforms.functional.hflip(image)
                if mask is not None:
                    mask = transforms.functional.hflip(mask)
            if random.random() < 0.5:
                image = transforms.functional.vflip(image)
                if mask is not None:
                    mask = transforms.functional.vflip(mask)
            angle = random.uniform(-180.0, 180.0)
            image = transforms.functional.rotate(
                image,
                angle,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=0,
            )
            if mask is not None:
                mask = transforms.functional.rotate(
                    mask,
                    angle,
                    interpolation=transforms.InterpolationMode.NEAREST,
                    fill=0,
                )

        if mask is None:
            return np.array(image), None
        return np.array(image), np.array(mask)

    def _normalize_ucmt_views(self, image_clean, mask=None):
        if self.training:
            image_student1 = np.array(
                self.ucmt_appearance(deepcopy(Image.fromarray(image_clean)))
            )
            image_student2 = np.array(
                self.ucmt_appearance(deepcopy(Image.fromarray(image_student1)))
            )
        else:
            image_student1 = image_clean
            image_student2 = image_clean

        if mask is None:
            clean = self.augmentation_1(image=image_clean)['image']
            student1 = self.augmentation_1(image=image_student1)['image']
            student2 = self.augmentation_1(image=image_student2)['image']
            return {
                'image': self.normalize_1(image=clean)['image'],
                'image_aug1': self.normalize_1(image=student1)['image'],
                'image_aug2': self.normalize_1(image=student2)['image'],
            }

        clean = self.augmentation_1(image=image_clean, mask=mask)
        student1 = self.augmentation_1(image=image_student1)['image']
        student2 = self.augmentation_1(image=image_student2)['image']
        clean_normalized = self.normalize_1(
            image=clean['image'],
            mask=clean['mask'],
        )
        return {
            'image': clean_normalized['image'],
            'image_aug1': self.normalize_1(image=student1)['image'],
            'image_aug2': self.normalize_1(image=student2)['image'],
            'mask': clean_normalized['mask'].long(),
        }

    def __getitem__(self, index):

        img_path_1 = self.img_paths_1[index]
        with Image.open(img_path_1) as image:
            img_1 = np.array(image.convert('RGB'))

        if self.sup:
            mask_path = self.mask_paths[index]
            with Image.open(mask_path) as mask_image:
                mask = np.array(mask_image.convert('L'))
            # FDCL is configured for two classes. Prepared masks use 0/255,
            # while segmentation losses require class indices 0/1.
            mask = (mask > 0).astype(np.uint8)

            if self.input_mode == 'ucmt_views':
                img_1, mask = self._shared_geometric_transform(img_1, mask)
                views = self._normalize_ucmt_views(img_1, mask)
                return {
                    'image': views['image'],
                    'image_aug1': views['image_aug1'],
                    'image_aug2': views['image_aug2'],
                    'mask': views['mask'],
                    'ID': os.path.split(mask_path)[1],
                }

            augment_1 = self.augmentation_1(image=img_1, mask=mask)
            img_1 = augment_1['image']
            mask_1 = augment_1['mask']
            stain_1 = self.stain_transform(img_1) if self.input_mode == 'rgb_stain' else None
            hed_1 = self.hed_transform(img_1) if self.input_mode == 'rgb_hed' else None
            normalize_1 = self.normalize_1(image=img_1, mask=mask_1)
            img_1 = normalize_1['image']
            mask_1 = normalize_1['mask']
            mask_1 = mask_1.long()

            sampel = {'image': img_1, 'mask': mask_1, 'ID': os.path.split(mask_path)[1]}
            if self.input_mode == 'rgb_stain':
                sampel['image_rgb'] = img_1
                sampel['image_stain'] = self.normalize_1(image=stain_1)['image']
            if self.input_mode == 'rgb_hed':
                sampel['image_rgb'] = img_1
                sampel['image_stain'] = hed_1

        else:
            if self.input_mode == 'ucmt_views':
                img_1, _ = self._shared_geometric_transform(img_1)
                views = self._normalize_ucmt_views(img_1)
                return {
                    'image': views['image'],
                    'image_aug1': views['image_aug1'],
                    'image_aug2': views['image_aug2'],
                    'ID': os.path.split(img_path_1)[1],
                }

            augment_1 = self.augmentation_1(image=img_1)
            img_1 = augment_1['image']
            stain_1 = self.stain_transform(img_1) if self.input_mode == 'rgb_stain' else None
            hed_1 = self.hed_transform(img_1) if self.input_mode == 'rgb_hed' else None
            normalize_1 = self.normalize_1(image=img_1)
            img_1 = normalize_1['image']

            sampel = {'image': img_1, 'ID': os.path.split(img_path_1)[1]}
            if self.input_mode == 'rgb_stain':
                sampel['image_rgb'] = img_1
                sampel['image_stain'] = self.normalize_1(image=stain_1)['image']
            if self.input_mode == 'rgb_hed':
                sampel['image_rgb'] = img_1
                sampel['image_stain'] = hed_1

        return sampel

    def __len__(self):
        return len(self.img_paths_1)


def imagefloder_itn(data_dir=None, input1='image', data_transform_1=None, data_normalize_1=None,
                    sup=True, num_images=None, manifest_path=None, input_mode='rgb_rgb',
                    training=True, **kwargs):
    dataset = dataset_itn(data_dir=data_dir,
                           input1=input1,
                           augmentation_1=data_transform_1,
                           normalize_1=data_normalize_1,
                           sup=sup,
                           num_images=num_images,
                           manifest_path=manifest_path,
                           input_mode=input_mode,
                           training=training,
                           **kwargs
                           )
    return dataset
