from torchsummary import summary
import torch.nn.functional as F

import torch
import torch.nn as nn


class MCNN(nn.Module):
    '''
    Implementation of MCNN
    '''

    def attrBranchCNN(self, use_gap):
        layers = [
            nn.Conv2d(in_channels=self.conv3_inch, out_channels=self.conv3_outch, kernel_size=(3, 3), padding=1,
                      bias=False),
            nn.BatchNorm2d(self.conv3_outch),
            nn.ReLU()
        ]
        if use_gap:
            layers.append(nn.AvgPool2d(self.conv3_in_size))
        else:
            layers.append(nn.MaxPool2d(kernel_size=(5, 5), stride=3))
            self.conv3_out_size = 18

        return torch.nn.Sequential(*layers)

    def attrBranchFC(self, num_attr, use_gap):
        if use_gap:
            fc1_in = self.conv3_outch
        else:
            self.conv3_out_size = 18
            fc1_in = self.conv3_out_size * self.conv3_out_size * self.conv3_outch

        fc_layers = [
            nn.Linear(fc1_in, 512),
            nn.ReLU(),
            nn.Dropout(0.50),

            nn.Linear(512, num_attr),
            nn.ReLU(),
            nn.Dropout(0.5)
        ]

        return torch.nn.Sequential(*fc_layers)

    def __init__(self):
        super(MCNN, self).__init__()

        self.use_gap = True

        '''
        Conv1 consists of 75 7x7 convolution filters, and it is followed by a ReLU, 3x3 Max Pooling, and 5x5 Normalization.
    
        will achieve 7x7 by 3 consecutive 3x3 blocks
        '''

        # 5x5 Normalization can be group normalization

        # Input image is 227x227
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=75, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(75),
            nn.ReLU(),

            nn.Conv2d(in_channels=75, out_channels=75, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(75),
            nn.ReLU(),

            nn.Conv2d(in_channels=75, out_channels=75, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(75),
            nn.ReLU(),

            # 3x3 max pooling with a stride of 2. Reference link: https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )  # Output 113x113x75

        '''
        Conv2 has 200 5x5 filters and it is also followed by a ReLU, 3x3 Max Pooling, and 5x5 Normalization
        '''
        self.conv2_inch = 75
        self.conv2_outch = 200
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv2_inch, out_channels=self.conv2_outch, kernel_size=(3, 3), padding=1,
                      bias=False),
            nn.BatchNorm2d(self.conv2_outch),
            nn.ReLU(),

            nn.Conv2d(in_channels=self.conv2_outch, out_channels=self.conv2_outch, kernel_size=(3, 3), padding=1,
                      bias=False),
            nn.BatchNorm2d(self.conv2_outch),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
        )  # Output 56x56x200

        '''
        Each Conv3 has 300 3x3 filters and is followed by a ReLU, 5x5 Max Pooling and 5x5 Normalization
    
        Each FC1 is fully connected to the corresponding previous layer
        Every FC1 has 512 units and is followed by a ReLU and a 50% dropout
    
        The FC2s are followed by a ReLU and a 50% dropout. Each FC2 is then fully connected to
        an output node for the attributes in that group
    
        '''
        self.conv3_inch = 200
        self.conv3_outch = 300
        self.conv3_in_size = 56
        self.conv3_out_size = 18

        # self.fc1_in = self.conv3_out_size * self.conv3_out_size * self.conv3_outch
        if self.use_gap:
            self.fc1_in = self.conv3_outch
        else:
            self.conv3_out_size = 18
            self.fc1_in = self.conv3_out_size * self.conv3_out_size * self.conv3_outch

        # Nose: Big Nose, Pointy Nose
        self.nose_attr = 2
        self.conv3_nose = self.attrBranchCNN(self.use_gap)
        self.fc_nose = self.attrBranchFC(self.nose_attr, self.use_gap)

        # Gender: Male
        self.gender_attr = 1
        self.conv3_gender = self.attrBranchCNN(self.use_gap)
        self.fc_gender = self.attrBranchFC(self.gender_attr, self.use_gap)

        # Mouth: Big Lips, Smiling, Lipstick, Mouth Slightly Open
        self.mouth_attr = 4
        self.conv3_mouth = self.attrBranchCNN(self.use_gap)
        self.fc_mouth = self.attrBranchFC(self.mouth_attr, self.use_gap)

        # Eyes: Arched Eyebrows, Bags Under Eyes, Bushy Eyebrows, Narrow Eyes, Eyeglasses
        self.eyes_attr = 5
        self.conv3_eyes = self.attrBranchCNN(self.use_gap)
        self.fc_eyes = self.attrBranchFC(self.eyes_attr, self.use_gap)

        # Face: Attractive, Blurry, Oval Face, Pale Skin, Young, Heavy Makeup
        self.face_attr = 6
        self.conv3_face = self.attrBranchCNN(self.use_gap)
        self.fc_face = self.attrBranchFC(self.face_attr, self.use_gap)

        # AroundHead(13): Black Hair, Blond Hair, Brown Hair, Gray Hair, Earrings,Necklace, Necktie, Balding, Receding Hairline, Bangs, Hat, Straight Hair, Wavy Hair
        # FacialHair(5): 5 o’clock Shadow, Mustache, No Beard, Sideburns, Goatee
        # Cheeks(2): High Cheekbones, Rosy Cheeks
        # Fat(2): Chubby, Double Chin

        self.head_attr = 13
        self.FacialHair_attr = 5
        self.cheeks_attr = 2
        self.fat_attr = 2

        self.conv3_rest = self.attrBranchCNN(self.use_gap)

        self.fc_head = self.attrBranchFC(self.head_attr, self.use_gap)

        self.fc_cheeks = self.attrBranchFC(self.cheeks_attr, self.use_gap)

        self.fc_FacialHair = self.attrBranchFC(self.FacialHair_attr, self.use_gap)

        self.fc_fat = self.attrBranchFC(self.fat_attr, self.use_gap)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)

        x = self.conv2(x)
        # print(x.shape)

        x_nose = self.conv3_nose(x)
        # print(x_nose.shape)

        x_eyes = self.conv3_eyes(x)
        x_face = self.conv3_face(x)
        x_gender = self.conv3_gender(x)
        x_mouth = self.conv3_mouth(x)
        x_rest = self.conv3_rest(x)

        x_rest = x_rest.view(-1, self.fc1_in)

        # Connect the FC for each group
        x_nose = self.fc_nose(x_nose.view(-1, self.fc1_in))
        x_eyes = self.fc_eyes(x_eyes.view(-1, self.fc1_in))
        x_face = self.fc_face(x_face.view(-1, self.fc1_in))
        x_gender = self.fc_gender(x_gender.view(-1, self.fc1_in))
        x_mouth = self.fc_mouth(x_mouth.view(-1, self.fc1_in))
        x_head = self.fc_head(x_rest)
        x_cheeks = self.fc_cheeks(x_rest)
        x_FacialHair = self.fc_FacialHair(x_rest)
        x_fat = self.fc_fat(x_rest)

        # sigmoid function
        # x_gender, x_nose, x_face, x_eyes, x_mouth, x_head, x_FacialHair, x_cheeks, x_fat

        out = {
            'gender': F.sigmoid(x_gender),
            'nose': F.sigmoid(x_nose),
            'mouth': F.sigmoid(x_mouth),
            'eyes': F.sigmoid(x_eyes),
            'face': F.sigmoid(x_face),
            'head': F.sigmoid(x_head),
            'facial_hair': F.sigmoid(x_FacialHair),
            'cheeks': F.sigmoid(x_cheeks),
            'fat': F.sigmoid(x_fat)}
        return out
