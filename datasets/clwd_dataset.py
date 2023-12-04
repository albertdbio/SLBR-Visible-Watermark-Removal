from PIL import Image
import numpy as np
import cv2
import os.path as osp
import os
import sys
import torch
from torchvision import datasets, transforms
from .base_dataset import get_transform
import random


class CLWDDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, args):
        args.is_train = is_train == "train"
        if args.is_train == True:
            self.root = args.dataset_dir + "/train"
            # self.keep_background_prob = 0.01
            self.keep_background_prob = -1
        elif args.is_train == False:
            self.root = args.dataset_dir + "/test"  #'/test/'
            self.keep_background_prob = -1
            # This overrides the cli arguments
            # args.preprocess = "resize"
            args.no_flip = True

        self.args = args
        # Augmentataion?
        self.transform_norm = transforms.Compose([transforms.ToTensor()])
        # transforms.Normalize(
        #         # (0.485, 0.456, 0.406),
        #         # (0.229, 0.224, 0.225)
        #     (0.5,0.5,0.5),
        #     (0.5,0.5,0.5)
        # )])
        self.augment_transform = get_transform(
            args,
            additional_targets={
                "J": "image",
                "I": "image",
                "watermark": "image",
                "mask": "mask",
                "alpha": "mask",
            },
        )  # ,
        self.transform_tensor = transforms.ToTensor()

        self.imageJ_path = osp.join(self.root, "Watermarked_image", "%s.jpg")
        self.imageI_path = osp.join(self.root, "Watermark_free_image", "%s.jpg")
        self.mask_path = osp.join(self.root, "Mask", "%s.png")
        # self.balance_path=osp.join(self.root,'Loss_balance','%s.png')
        self.alpha_path = osp.join(self.root, "Alpha", "%s.png")
        self.W_path = osp.join(self.root, "Watermark", "%s.png")

        self.ids = list()
        resolved_path = osp.realpath(self.root + "/Watermarked_image")
        for file in os.listdir(resolved_path):
            self.ids.append(file.strip(".jpg"))        
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

    def __len__(self):
        return len(self.ids)

    # Used for debugging
    def generate_image(self, height, width):
        # create a random tensor of 256 x 256 x 3
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    # Used for debugging
    def generate_mask(self, height, width):
        mask = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        return mask

    def get_sample(self, index):
        # Get the input_height and input_width from the args
        input_height = self.args.input_height
        input_width = self.args.input_width

        # Print the image id, which is the filename without the extension
        img_id = self.ids[index]
        # print(img_id)
        # img_id = self.corrupt_list[index % len(self.corrupt_list)].split('.')[0]

        # img_J = cv2.imread(self.imageJ_path%img_id)
        img_J = self.generate_image(input_height, input_width)

        # img_I = cv2.imread(self.imageI_path%img_id)        
        img_I = self.generate_image(input_height, input_width)

        # w = cv2.imread(self.W_path%img_id)        
        w = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        if w is None:
            print(self.W_path % img_id)
        w = cv2.cvtColor(w, cv2.COLOR_BGR2RGB)

        # mask = cv2.imread(self.mask_path%img_id)        
        mask = self.generate_mask(input_height, input_width)

        # alpha = cv2.imread(self.alpha_path%img_id)        
        alpha = self.generate_mask(input_height, input_width)

        return {
            "J": img_J,
            "I": img_I,
            "watermark": w,
            "mask": mask,
            "alpha": alpha,
            "img_path": self.imageJ_path % img_id,
        }

    def __getitem__(self, index):
        sample = self.get_sample(index)
        self.check_sample_types(sample)
        sample = self.augment_sample(sample)

        J = self.transform_norm(sample["J"])
        I = self.transform_norm(sample["I"])
        w = self.transform_norm(sample["watermark"])

        mask = sample["mask"][np.newaxis, ...].astype(np.float32)
        mask = np.where(mask > 0.1, 1, 0).astype(np.uint8)
        alpha = sample["alpha"][np.newaxis, ...].astype(np.float32)

        data = {
            "image": J,
            "target": I,
            "wm": w,
            "mask": mask,
            "alpha": alpha,
            "img_path": sample["img_path"],
        }
        return data

    def check_sample_types(self, sample):
        assert sample["J"].dtype == "uint8"
        assert sample["I"].dtype == "uint8"
        assert sample["watermark"].dtype == "uint8"

    def augment_sample(self, sample):
        if self.augment_transform is None:
            return sample
        # print(self.transform.additional_targets.keys())
        additional_targets = {
            target_name: sample[target_name]
            for target_name in self.augment_transform.additional_targets.keys()
        }

        valid_augmentation = False
        while not valid_augmentation:
            aug_output = self.augment_transform(image=sample["I"], **additional_targets)
            valid_augmentation = self.check_augmented_sample(sample, aug_output)

        for target_name, transformed_target in aug_output.items():
            # print(target_name,transformed_target.shape)
            sample[target_name] = transformed_target

        return sample

    def check_augmented_sample(self, sample, aug_output):
        if (
            self.keep_background_prob < 0.0
            or random.random() < self.keep_background_prob
        ):
            return True
        return aug_output["mask"].sum() > 100
