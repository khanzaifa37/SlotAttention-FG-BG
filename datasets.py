import os
import glob
import json
import random
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from pycocotools import mask
from pycocotools.coco import COCO
from utils_spot import GaussianBlur

ImageFile.LOAD_TRUNCATED_IMAGES = True


def resolve_coco_image_dir(root, split, year='2017', image_root=None):
    search_roots = []
    if image_root is not None:
        search_roots.append(image_root)
    search_roots.extend([root, os.path.join(root, "images")])

    for candidate_root in search_roots:
        candidate = os.path.join(candidate_root, f"{split}{year}")
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find COCO image directory for split '{split}{year}'. "
        f"Searched under: {search_roots}"
    )


@dataclass
class TeacherView:
    crop: torch.Tensor
    coords: torch.Tensor
    flipped: torch.Tensor


class TeacherPairAugmentation:
    def __init__(self, image_size=224, min_scale=0.08, max_scale=1.0):
        self.image_size = image_size
        self.scale = (min_scale, max_scale)
        self.ratio = (3.0 / 4.0, 4.0 / 3.0)
        self.color_jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        self.blur = GaussianBlur(p=1.0)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def _make_view(self, image):
        width, height = image.size
        top, left, crop_h, crop_w = transforms.RandomResizedCrop.get_params(
            image, scale=self.scale, ratio=self.ratio
        )

        crop = TF.resized_crop(
            image,
            top=top,
            left=left,
            height=crop_h,
            width=crop_w,
            size=(self.image_size, self.image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )

        flipped = False
        if random.random() < 0.5:
            crop = TF.hflip(crop)
            flipped = True

        if random.random() < 0.8:
            crop = self.color_jitter(crop)
        if random.random() < 0.2:
            crop = TF.rgb_to_grayscale(crop, num_output_channels=3)
        if random.random() < 0.5:
            crop = self.blur(crop)

        coords = torch.tensor(
            [
                left / width,
                top / height,
                (left + crop_w) / width,
                (top + crop_h) / height,
            ],
            dtype=torch.float32,
        )

        return TeacherView(
            crop=self.to_tensor(crop),
            coords=coords,
            flipped=torch.tensor(flipped, dtype=torch.bool),
        )

    def __call__(self, image):
        view_1 = self._make_view(image)
        view_2 = self._make_view(image)
        return (
            (view_1.crop, view_2.crop),
            (view_1.coords, view_2.coords),
            (view_1.flipped, view_2.flipped),
        )


class PascalVOC(Dataset):
    def __init__(self, root, split, image_size=224, mask_size = 224):
        assert split in ['trainaug', 'val',"val_viz"]
        imglist_fp = os.path.join(root, 'ImageSets/Segmentation', split+'.txt')
        self.imglist = self.read_imglist(imglist_fp)

        self.root = root
        self.train_transform = transforms.Compose([
                            transforms.Resize(size=image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                            transforms.RandomCrop(image_size),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        ])

        self.val_transform_image = transforms.Compose([transforms.Resize(size = image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                               transforms.CenterCrop(size = image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.val_transform_mask = transforms.Compose([transforms.Resize(size = mask_size, interpolation=transforms.InterpolationMode.NEAREST),
                               transforms.CenterCrop(size = mask_size),
                               transforms.PILToTensor()])
        self.split = split
        self.image_size = image_size
        self.mask_size = mask_size

    def __getitem__(self, idx):

        imgname = self.imglist[idx]
        img_fp = os.path.join(self.root, 'JPEGImages', imgname) + '.jpg'
        # img_fp = os.path.join("/data1/ysj/VOCdevkit/stage1_fg", imgname) + '.jpg'
        mask_fp_class = os.path.join(self.root, 'SegmentationClass', imgname) + '.png'
        mask_fp_instance = os.path.join(self.root, 'SegmentationObject', imgname) + '.png'

        img = Image.open(img_fp)

        if self.split=='trainaug':           
            img = self.train_transform(img)
            
            return img
   
        elif self.split=='val':
            
            mask_class    = Image.open(mask_fp_class)
            mask_instance = Image.open(mask_fp_instance)
            
            img = self.val_transform_image(img)
            
            mask_class = self.val_transform_mask(mask_class).squeeze().long()
            mask_class[mask_class==255]=0 # Ignore objects' boundaries

            mask_instance = self.val_transform_mask(mask_instance).squeeze().long()
            mask_instance[mask_instance==255]=0 # Ignore objects' boundaries
            
            ignore_mask = torch.zeros((1,self.mask_size,self.mask_size), dtype=torch.long) # There is no overlapping in VOC

            return img, mask_instance, mask_class, ignore_mask
        
        elif self.split=='val_viz':
            
            mask_class    = Image.open(mask_fp_class)
            mask_instance = Image.open(mask_fp_instance)
            
            img = self.val_transform_image(img)
            
            mask_class = self.val_transform_mask(mask_class).squeeze().long()
            mask_class[mask_class==255]=0 # Ignore objects' boundaries

            mask_instance = self.val_transform_mask(mask_instance).squeeze().long()
            mask_instance[mask_instance==255]=0 # Ignore objects' boundaries
            
            ignore_mask = torch.zeros((1,self.mask_size,self.mask_size), dtype=torch.long) # There is no overlapping in VOC

            return img, mask_instance, mask_class, ignore_mask
        
        else:
            
            mask_class    = Image.open(mask_fp_class)
            mask_instance = Image.open(mask_fp_instance)
            
            return img, mask_instance.long(), mask_instance.squeeze()
        



    def __len__(self):
        return len(self.imglist)

    def read_imglist(self, imglist_fp):
        ll = []
        with open(imglist_fp, 'r') as fd:
            for line in fd:
                ll.append(line.strip())
        return ll


class PascalVOCTeacher(Dataset):
    def __init__(self, root, split='trainaug', image_size=224, min_scale=0.08):
        assert split in ['trainaug', 'val']
        imglist_fp = os.path.join(root, 'ImageSets/Segmentation', split + '.txt')
        self.imglist = []
        with open(imglist_fp, 'r') as fd:
            for line in fd:
                self.imglist.append(line.strip())
        self.root = root
        self.transform = TeacherPairAugmentation(image_size=image_size, min_scale=min_scale)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        imgname = self.imglist[idx]
        img_fp = os.path.join(self.root, 'JPEGImages', imgname) + '.jpg'
        img = Image.open(img_fp).convert('RGB')
        return self.transform(img)


class COCO2017(Dataset):
    NUM_CLASSES = 81
    CAT_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 
 89, 90]
    
    assert(NUM_CLASSES) == len(set(CAT_LIST))

    def __init__(self, root, split='train', year='2017', image_size=224, mask_size=224, return_gt_in_train=False, image_root=None):
        super().__init__()
        ann_file = os.path.join(root, 'annotations/instances_{}{}.json'.format(split, year))
        self.img_dir = resolve_coco_image_dir(root, split, year=year, image_root=image_root)
        self.split = split
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        self.return_gt_in_train = return_gt_in_train

        self.ids = list(self.coco.imgs.keys())
        
        self.train_transform = transforms.Compose([
                            transforms.Resize(size=image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                            transforms.CenterCrop(image_size),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        ])
        
        self.val_transform_image = transforms.Compose([transforms.Resize(size = image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                               transforms.CenterCrop(size = image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.val_transform_mask = transforms.Compose([transforms.Resize(size = mask_size, interpolation=transforms.InterpolationMode.NEAREST),
                               transforms.CenterCrop(size = mask_size),
                               transforms.PILToTensor()])
        self.image_size = image_size

    def __getitem__(self, index):
        img, mask_instance, mask_class, mask_ignore = self._make_img_gt_point_pair(index)

        if self.split == "train" and (self.return_gt_in_train is False):
            
            img = self.train_transform(img)
            
            return img
        elif self.split == "train" and (self.return_gt_in_train is True):
            img = self.val_transform_image(img)
            mask_class = self.val_transform_mask(mask_class)
            mask_instance = self.val_transform_mask(mask_instance)
            mask_ignore = self.val_transform_mask(mask_ignore)

            if random.random() < 0.5:
                img = TF.hflip(img)
                mask_class = TF.hflip(mask_class)
                mask_instance = TF.hflip(mask_instance)
                mask_ignore = TF.hflip(mask_ignore)
            
            mask_class = mask_class.squeeze().long()
            mask_instance = mask_instance.squeeze().long()
            mask_ignore = mask_ignore.squeeze().long()

            return img, mask_instance, mask_class, mask_ignore        
        elif self.split =='val':

            img = self.val_transform_image(img)
            mask_class = self.val_transform_mask(mask_class).squeeze().long()
            mask_instance = self.val_transform_mask(mask_instance).squeeze().long()
            mask_ignore = self.val_transform_mask(mask_ignore).squeeze().long().unsqueeze(0)
            
            return img, mask_instance, mask_class, mask_ignore
        else:
            raise

    def _make_img_gt_point_pair(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        _targets = self._gen_seg_n_insta_masks(cocotarget, img_metadata['height'], img_metadata['width'])
        mask_class = Image.fromarray(_targets[0])
        mask_instance = Image.fromarray(_targets[1])
        mask_ignore = Image.fromarray(_targets[2])
        return _img, mask_instance, mask_class, mask_ignore

    def _gen_seg_n_insta_masks(self, target, h, w):
        seg_mask = np.zeros((h, w), dtype=np.uint8)
        insta_mask = np.zeros((h, w), dtype=np.uint8)
        ignore_mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for i, instance in enumerate(target, 1):
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                seg_mask[:, :] += (seg_mask == 0) * (m * c)
                insta_mask[:, :] += (insta_mask == 0) * (m * i)
                ignore_mask[:, :] += m
            else:
                seg_mask[:, :] += (seg_mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
                insta_mask[:, :] += (insta_mask == 0) * (((np.sum(m, axis=2)) > 0) * i).astype(np.uint8)
                ignore_mask[:, :] += (((np.sum(m, axis=2)) > 0) * 1).astype(np.uint8)

        # Ignore overlaps
        ignore_mask = (ignore_mask>1).astype(np.uint8)

        all_masks = np.stack([seg_mask, insta_mask, ignore_mask])
        return all_masks

    def __len__(self):
        return len(self.ids)


class COCO2017Teacher(Dataset):
    def __init__(self, root, split='train', year='2017', image_size=224, min_scale=0.08, image_root=None):
        ann_file = os.path.join(root, 'annotations/instances_{}{}.json'.format(split, year))
        self.img_dir = resolve_coco_image_dir(root, split, year=year, image_root=image_root)
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transform = TeacherPairAugmentation(image_size=image_size, min_scale=min_scale)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_metadata = self.coco.loadImgs(img_id)[0]
        image = Image.open(os.path.join(self.img_dir, img_metadata['file_name'])).convert('RGB')
        return self.transform(image)


class MOVi(Dataset):
    def __init__(self, root, split, image_size, mask_size, num_segs=25, frames_per_clip=24, img_glob='*_image.png', predefined_json_paths = None):
        
        self.root = root
        self.split = split
        self.image_size = image_size
        self.mask_size = mask_size
        self.total_dirs = sorted(glob.glob(os.path.join(root, '*')))
        self.frames_per_clip = frames_per_clip
        
        if self.split == 'train' and predefined_json_paths is not None:
            with open(predefined_json_paths, 'r') as fp:
                paths_persistence = json.load(fp)
            self.rgb = [Path(p) for p in paths_persistence['rgb']]
            self.mask = [[Path(p) for p in m] for m in paths_persistence['mask']]
            
        else:
            self.rgb = []
            self.mask = []
            for dir in self.total_dirs:
                frame_buffer = []
                mask_buffer = []
                image_paths = glob.glob(os.path.join(dir, img_glob))
                if self.split == 'train':
                    random.shuffle(image_paths)
                    image_paths = image_paths[:self.frames_per_clip]
                else:
                    image_paths = sorted(image_paths)
                for image_path in image_paths:
                    p = Path(image_path)
    
                    frame_buffer.append(p)
                    mask_buffer.append([
                        p.parent / f"{p.stem.split('_')[0]}_mask_{n:02}.png" for n in range(num_segs)
                    ])
    
                self.rgb.extend(frame_buffer)
                self.mask.extend(mask_buffer)
                frame_buffer = []
                mask_buffer = []
            
        if self.split == 'train' and predefined_json_paths is None:
            paths_persistence = dict(rgb=[str(p) for p in self.rgb], mask=[[str(p) for p in m] for m in self.mask])
                    
            with open(self.split+'_movi_paths.json', 'w') as fp:
                json.dump(paths_persistence, fp)
        
        self.train_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), 
                                                                  (0.229, 0.224, 0.225))])
        self.val_transforms = transforms.ToTensor()

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, idx):
        
        img_loc = self.rgb[idx]
        img = Image.open(img_loc).convert("RGB")
        img = img.resize((self.image_size, self.image_size))
        img = self.train_transform(img)

        if self.split == 'train':
            return img
        else:
            mask_locs = self.mask[idx]
            masks = []
            for mask_loc in mask_locs:
                mask = Image.open(mask_loc).convert('1')
                mask = mask.resize((self.mask_size, self.mask_size))
                mask = self.val_transforms(mask)
                masks += [mask]
            masks = torch.stack(masks, dim=0).squeeze().long()
    
            mask_instance = torch.zeros((self.mask_size,self.mask_size), dtype=torch.long)
            mask_class = torch.zeros((self.mask_size,self.mask_size), dtype=torch.long) # There are no semantic segmentations in MOVi
            ignore_mask = torch.zeros((1,self.mask_size,self.mask_size), dtype=torch.long) # There is no overlapping in MOVi
            
            for i, instance in enumerate(masks):
                mask_instance[:, :] += instance * i

            return img, mask_instance, mask_class, ignore_mask


class TumorDataset(Dataset):
    """
    Generic dataset for tumor segmentation images.

    Expected directory layout
    -------------------------
    images_dir/   img001.png  img002.jpg  ...
    masks_dir/    img001.png  img002.png  ...   (same stem, any extension)

    Mask conventions (handled automatically)
    ----------------------------------------
    - Binary masks  (0 / 255  or  0 / 1):
        Each connected component → separate instance.
    - Soft-boundary masks (e.g. BRISC: 0=bg, 1-7=uncertain boundary, 248-254=near-edge, 255=tumor):
        Pixels >= mask_threshold are treated as tumor.
        Connected components of that region → separate instances.
    - Instance masks (0=bg, 1..N=distinct instances):
        Used directly as-is.

    mask_threshold : pixels with value >= this are treated as foreground.
                     Default 1 (any non-zero pixel).  Set to 128 or 255 to
                     keep only the high-confidence tumor core (recommended
                     for BRISC-style soft-boundary masks).

    Training split  → image tensor only  (unsupervised — no masks used)
    Validation split → (image, mask_instance, mask_class, ignore_mask)
        mask_instance : H×W long,   0=background  1..N=instance IDs
        mask_class    : H×W long,   0=background  1=tumor
        ignore_mask   : 1×H×W long, all zeros
    """

    def __init__(self, images_dir, masks_dir=None, split='train',
                 image_size=224, mask_size=224, mask_ext='.png',
                 mask_threshold=1):
        assert split in ('train', 'val')
        if split == 'val':
            assert masks_dir is not None, 'masks_dir is required for the val split'

        self.split = split
        self.image_size = image_size
        self.mask_size = mask_size
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.mask_ext = mask_ext
        self.mask_threshold = mask_threshold

        img_exts = ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff')
        self.image_paths = sorted(
            p for ext in img_exts
            for p in glob.glob(os.path.join(images_dir, ext))
        )
        assert len(self.image_paths) > 0, f'No images found in {images_dir}'

        self.train_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.val_transform_image = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.val_transform_mask = transforms.Compose([
            transforms.Resize(mask_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(mask_size),
            transforms.PILToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def _load_instance_mask(self, mask_path):
        """
        Load a mask PNG and return a 2-D int32 array where each unique
        positive integer is a separate tumor instance (0 = background).

        Decision logic
        --------------
        1. Apply self.mask_threshold: pixels >= threshold are foreground.
        2. If the foreground contains small consecutive ints (1..N, max<=200)
           treat them as pre-labeled instance IDs.
        3. Otherwise (binary or BRISC soft-boundary masks where core=255,
           fringe=1-7/248-254): binarize at threshold and run
           connected-component labeling — each contiguous blob = one instance.
        """
        from scipy import ndimage as ndi

        mask_arr = np.array(Image.open(mask_path).convert('L'), dtype=np.int32)

        foreground = (mask_arr >= self.mask_threshold)

        if not foreground.any():
            return np.zeros_like(mask_arr)

        unique_pos = np.unique(mask_arr[foreground])

        # Pre-labeled instance mask: small consecutive non-zero integers
        if unique_pos.max() <= 200 and unique_pos.min() >= 1:
            return np.where(foreground, mask_arr, 0).astype(np.int32)

        # Binary or soft-boundary mask: binarize then label components
        labeled, _ = ndi.label(foreground.astype(np.uint8))
        return labeled.astype(np.int32)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        if self.split == 'train':
            return self.train_transform(img)

        # ----- validation -----
        stem = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.masks_dir, stem + self.mask_ext)

        instance_arr = self._load_instance_mask(mask_path)
        # Class mask: all tumors share class label 1
        class_arr = (instance_arr > 0).astype(np.int32)

        img_t = self.val_transform_image(img)

        # PIL conversion: clip to uint8 range (max 255 instances per image)
        instance_arr = np.clip(instance_arr, 0, 255).astype(np.uint8)
        class_arr    = class_arr.astype(np.uint8)

        mask_instance = self.val_transform_mask(Image.fromarray(instance_arr)).squeeze().long()
        mask_class    = self.val_transform_mask(Image.fromarray(class_arr)).squeeze().long()
        ignore_mask   = torch.zeros((1, self.mask_size, self.mask_size), dtype=torch.long)

        return img_t, mask_instance, mask_class, ignore_mask


class TumorDatasetTeacher(Dataset):
    """
    Teacher-training split for tumor images.

    Applies TeacherPairAugmentation (two paired random crops per image) so
    the Indicator model can learn foreground vs background from unlabeled data.
    No masks are required — teacher training is fully unsupervised.

    Expected layout
    ---------------
    images_dir/   img001.png  img002.jpg  ...

    Returns the same (crops, coords, flags) tuple as PascalVOCTeacher /
    COCO2017Teacher so it drops in directly to train_teacher.py.
    """

    def __init__(self, images_dir, image_size=224, min_scale=0.08):
        img_exts = ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff')
        self.image_paths = sorted(
            p for ext in img_exts
            for p in glob.glob(os.path.join(images_dir, ext))
        )
        assert len(self.image_paths) > 0, f'No images found in {images_dir}'
        self.transform = TeacherPairAugmentation(image_size=image_size,
                                                 min_scale=min_scale)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)
