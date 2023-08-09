import json
import os
import random

from torch.utils.data import Dataset
from collections import defaultdict

from PIL import Image
from PIL import ImageFile
from dataset.utils import pre_caption

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None




class re_split_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, config, split, max_words=30):
        self.ann_imgs = json.load(open(ann_file, 'r'))['images']
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        # mapping total file to split file
        self.img_order2id = {}
        self.img_id2order = {}
        self.text_order2id = {}
        self.text_id2order = {}
        self.ann = []
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_order = 0
        img_order = 0
        for img in self.ann_imgs:
            if split == 'all':
                assert img['split'] in ['train', 'val', 'test']
                prefix = img['filepath'] if 'filepath' in img else config['image_prefix'][img['split']]
            elif img['split'] == split:
                prefix = img['filepath'] if 'filepath' in img else config['image_prefix'][split]
            else:
                continue

            prefix_image_name = os.path.join(prefix, img['filename'])
            self.image.append(prefix_image_name)
            self.img2txt[img_order] = []
            self.img_id2order[img['imgid']] = img_order
            self.img_order2id[img_order] = img['imgid']
            caps = []
            for caption in img['sentences']:
                caps.append(caption['raw'])
                self.text.append(pre_caption(caption['raw'], self.max_words))
                self.img2txt[img_order].append(txt_order)
                self.txt2img[txt_order] = img_order
                self.text_id2order[caption['sentid']] = txt_order
                self.text_order2id[txt_order] = caption['sentid']
                txt_order += 1
            self.ann.append({'image': img['filename'],
                             'caption': caps})
            img_order += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        prefix_img_name = self.image[index]
        image_path = os.path.join(self.image_root, prefix_img_name)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index


class re_nocaps_eval_dataset(Dataset):
    def __init__(self, config, transform, max_words=30):
        self.ann = json.load(open(config['split_file'], 'r'))
        self.transform = transform
        self.image_root = config['image_root']
        self.max_words = max_words

        self.img_order2id = {}
        self.img_id2order = {}
        self.text_order2id = {}
        self.text_id2order = {}

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = defaultdict(list)

        img_order = 0
        for image in self.ann['images']:
            if image['domain'] in config['data_type']:
                self.image.append(image['file_name'])
                self.img_id2order[image['id']] = img_order
                self.img_order2id[img_order] = image['id']
                img_order += 1

        text_order = 0
        for text in self.ann['annotations']:
            if text['image_id'] in self.img_id2order.keys():
                self.text.append(pre_caption(text['caption'], self.max_words))
                img_order = self.img_id2order[text['image_id']]
                self.img2txt[img_order].append(text_order)
                self.txt2img[text_order] = img_order

                self.text_id2order[text['id']] = text_order
                self.text_order2id[text_order] = text['id']
                text_order += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        prefix_img_name = self.image[index]
        image_path = os.path.join(self.image_root, prefix_img_name)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index


