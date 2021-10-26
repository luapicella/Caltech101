from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        self.images_dataset = []
        self.labels = []
        self.labels_indx = []
        self.unique_labels = []

        file_path = "Caltech101" + "/" + self.split + ".txt"
        print (file_path)
        with open(file_path, "r" ) as fp:
            for line in fp:
                row = line.strip("\n")
                label_tmp = row.split("/")[0]
                if (label_tmp != "BACKGROUND_Google"):
                    img = pil_loader(root + "/" + row)
                    self.images_dataset.append(img)
                    self.labels.append(label_tmp)

        self.unique_labels = list(dict.fromkeys(self.labels))
        for lab in self.labels:
            self.labels_indx.append(list(self.unique_labels).index(lab))

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label =   self.images_dataset[index], self.labels_indx[index]

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.images_dataset)
        return length
