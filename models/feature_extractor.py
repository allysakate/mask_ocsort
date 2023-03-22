import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


def load_model(model, model_file):
    checkpoint = torch.load(model_file)
    # Support for checkpoints saved by scripts based off of
    #   https://github.com/pytorch/examples/blob/master/imagenet/main.py
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=False)


class AlexNetPartial(nn.Module):
    supported_layers = [
        "conv1",
        "conv2",
        "conv3",
        "conv4",
        "conv5",
        "fc6",
        "relu6",
        "fc7",
        "fc8",
    ]

    def __init__(self, layer, model_file=None, data_parallel=False, **kwargs):
        super(AlexNetPartial, self).__init__()
        assert layer in AlexNetPartial.supported_layers
        self.model = models.alexnet(**kwargs)

        self.output_layer = layer

        if "conv" in self.output_layer:
            # Map, e.g., 'conv2' to corresponding index into self.features
            conv_map = {}
            conv_index = 1
            for i, layer in enumerate(self.model.features):
                if isinstance(layer, nn.Conv2d):
                    conv_map["conv%s" % conv_index] = i
                    conv_index += 1
            requested_index = conv_map[self.output_layer]
            features = list(self.model.features.children())[: requested_index + 1]
            self.model.features = nn.Sequential(*features)
        else:
            classifier_map = {"fc6": 1, "relu6": 2, "fc7": 4, "relu7": 5, "fc8": 6}
            requested_index = classifier_map[self.output_layer]
            classifier = list(self.model.classifier.children())[: requested_index + 1]
            self.model.classifier = nn.Sequential(*classifier)
        if data_parallel:
            self.model.features = torch.nn.DataParallel(self.model.features)
        if model_file is not None:
            load_model(self.model, model_file)

    def forward(self, x):
        x = self.model.features(x)
        if "conv" in self.output_layer:
            return x
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.model.classifier(x)
        return x


# https://ieeexplore-ieee-org.dlsu.idm.oclc.org/stamp/stamp.jsp?tp=&arnumber=9606052&tag=1
class FeatExtractor:
    def __init__(self, arch_layer="alexnet-fc7", use_cuda=True):
        partial_models = {
            "alexnet": AlexNetPartial,
        }
        arch_layer = "alexnet-fc7"
        architecture, layer = arch_layer.split("-")

        construction_kwargs = {
            "layer": layer,
            "pretrained": True,
            "model_file": None,
            "data_parallel": True,
        }
        if architecture.startswith("densenet") or architecture.startswith("vgg"):
            construction_kwargs["architecture"] = architecture
        self.model = partial_models[architecture](**construction_kwargs)
        self.model.cuda()
        self.model.eval()

    def _preprocess(self, im_crops):
        transform_image = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        color_converted = cv2.cvtColor(im_crops, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_converted)
        return transform_image(pil_image).unsqueeze(0)

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            input_var = torch.autograd.Variable(im_batch).cuda()
            features = self.model(input_var).data.cpu().numpy()[0]
        return features


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, is_downsample=False):
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False), nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False), nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y), True)


def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [
                BasicBlock(c_in, c_out, is_downsample=is_downsample),
            ]
        else:
            blocks += [
                BasicBlock(c_out, c_out),
            ]
    return nn.Sequential(*blocks)


class Net(nn.Module):
    def __init__(self, num_classes=751, reid=False):
        super(Net, self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(64, 64, 2, False)
        # 32 64 32
        self.layer2 = make_layers(64, 128, 2, True)
        # 64 32 16
        self.layer3 = make_layers(128, 256, 2, True)
        # 128 16 8
        self.layer4 = make_layers(256, 512, 2, True)
        # 256 8 4
        self.avgpool = nn.AvgPool2d((8, 4), 1)
        # 256 1 1
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # B x 128
        if self.reid:
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        # classifier
        x = self.classifier(x)
        return x


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)[
            "net_dict"
        ]
        self.net.load_state_dict(state_dict)
        print("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """

        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255.0, size)

        im_batch = torch.cat(
            [self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0
        ).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == "__main__":
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)
