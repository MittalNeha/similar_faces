import os
import pandas as pd
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


generalFolder = '/content/'
allAttributes = pd.read_csv(generalFolder + 'list_attr_celeba.csv')

'''
The face attributes have been grouped into 9 groups

'''
attr_names = list(allAttributes.columns[1:])
groups = [ ['Male',],
          ['Big_Nose', 'Pointy_Nose'],
          ['Big_Lips', 'Smiling',  'Mouth_Slightly_Open', 'Wearing_Lipstick'],
          ['Arched_Eyebrows', 'Bags_Under_Eyes', 'Bushy_Eyebrows', 'Narrow_Eyes', 'Eyeglasses'],
          ['Attractive', 'Blurry', 'Oval_Face', 'Pale_Skin', 'Young', 'Heavy_Makeup'],
          ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'Bald', 'Receding_Hairline', 'Bangs','Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Necklace', 'Wearing_Necktie', 'Wearing_Hat',],
          ['5_o_Clock_Shadow', 'Mustache', 'No_Beard', 'Sideburns', 'Goatee'],
          ['High_Cheekbones', 'Rosy_Cheeks'],
          ['Chubby', 'Double_Chin']
          ]


def get_attr_groups(attributes, transform=None):
    attr_groups = []

    for at in groups:

        b = [attributes[attr_names.index(a)] for a in at]
        attr_groups.append(b)
    # print(attr_groups)
    if transforms:
        return {
            'gender': torch.tensor(attr_groups[0]),
            'nose': torch.tensor(attr_groups[1]),
            'mouth': torch.tensor(attr_groups[2]),
            'eyes': torch.tensor(attr_groups[3]),
            'face': torch.tensor(attr_groups[4]),
            'head': torch.tensor(attr_groups[5]),
            'facial_hair': torch.tensor(attr_groups[6]),
            'cheeks': torch.tensor(attr_groups[7]),
            'fat': torch.tensor(attr_groups[8])
        }
    else:
        return {
            'gender': attr_groups[0],
            'nose': attr_groups[1],
            'mouth': attr_groups[2],
            'eyes': attr_groups[3],
            'face': attr_groups[4],
            'head': attr_groups[5],
            'facial_hair': attr_groups[6],
            'cheeks': attr_groups[7],
            'fat': attr_groups[8]}


class FaceAttributesDataset(Dataset):
    '''
    Reading the attributes from CelebA dataset
    '''

    def __init__(self, root_dir, img_dir, attr_csv, transform=None):
        self.img_dir = root_dir + img_dir
        self.attributes = pd.read_csv(root_dir + attr_csv)
        self.transform = transform

    def __len__(self):
        return len(self.attributes)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.attributes.iloc[idx, 0])
        image = Image.open(img_name)

        image = image.resize((227, 227))
        # Change attribute range from 0 to 1
        attributes = get_attr_groups(list((self.attributes.iloc[idx, 1:] + 1) / 2), self.transform)

        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'attr': attributes}
        return sample
