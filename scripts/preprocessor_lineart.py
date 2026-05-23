from modules_forge.supported_preprocessor import Preprocessor, PreprocessorParameter
from modules_forge.shared import add_supported_preprocessor, preprocessor_dir
from modules import devices
from modules_forge.utils import resize_image_with_pad, HWC3
from modules.modelloader import load_file_from_url

import torch
import numpy
import cv2
import os
from einops import rearrange


## manga (anime_denoised)
## MIT License: Copyright (c) 2021 Miaomiao Li
class _bn_relu_conv(torch.nn.Module):
    def __init__(self, in_filters, nb_filters, fw, fh, subsample=1):
        super(_bn_relu_conv, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_filters, eps=1e-3),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(in_filters, nb_filters, (fw, fh), stride=subsample, padding=(fw//2, fh//2), padding_mode='zeros')
        )

    def forward(self, x):
        return self.model(x)


class _u_bn_relu_conv(torch.nn.Module):
    def __init__(self, in_filters, nb_filters, fw, fh, subsample=1):
        super(_u_bn_relu_conv, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_filters, eps=1e-3),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(in_filters, nb_filters, (fw, fh), stride=subsample, padding=(fw//2, fh//2)),
            torch.nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        return self.model(x)


class _shortcut(torch.nn.Module):
    def __init__(self, in_filters, nb_filters, subsample=1):
        super(_shortcut, self).__init__()
        self.process = False
        self.model = None
        if in_filters != nb_filters or subsample != 1:
            self.process = True
            self.model = torch.nn.Sequential(
                    torch.nn.Conv2d(in_filters, nb_filters, (1, 1), stride=subsample)
                )

    def forward(self, x, y):
        if self.process:
            y0 = self.model(x)
            return y0 + y
        else:
            return x + y


class _u_shortcut(torch.nn.Module):
    def __init__(self, in_filters, nb_filters, subsample):
        super(_u_shortcut, self).__init__()
        self.process = False
        self.model = None
        if in_filters != nb_filters:
            self.process = True
            self.model = torch.nn.Sequential(
                torch.nn.Conv2d(in_filters, nb_filters, (1, 1), stride=subsample, padding_mode='zeros'),
                torch.nn.Upsample(scale_factor=2, mode='nearest')
            )

    def forward(self, x, y):
        if self.process:
            return self.model(x) + y
        else:
            return x + y


class basic_block(torch.nn.Module):
    def __init__(self, in_filters, nb_filters, init_subsample=1):
        super(basic_block, self).__init__()
        self.conv1 = _bn_relu_conv(in_filters, nb_filters, 3, 3, subsample=init_subsample)
        self.residual = _bn_relu_conv(nb_filters, nb_filters, 3, 3)
        self.shortcut = _shortcut(in_filters, nb_filters, subsample=init_subsample)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.residual(x1)
        return self.shortcut(x, x2)


class _u_basic_block(torch.nn.Module):
    def __init__(self, in_filters, nb_filters, init_subsample=1):
        super(_u_basic_block, self).__init__()
        self.conv1 = _u_bn_relu_conv(in_filters, nb_filters, 3, 3, subsample=init_subsample)
        self.residual = _bn_relu_conv(nb_filters, nb_filters, 3, 3)
        self.shortcut = _u_shortcut(in_filters, nb_filters, subsample=init_subsample)

    def forward(self, x):
        y = self.residual(self.conv1(x))
        return self.shortcut(x, y)


class _residual_block(torch.nn.Module):
    def __init__(self, in_filters, nb_filters, repetitions, is_first_layer=False):
        super(_residual_block, self).__init__()
        layers = []
        for i in range(repetitions):
            init_subsample = 1
            if i == repetitions - 1 and not is_first_layer:
                init_subsample = 2

            if i == 0:
                layer = basic_block(in_filters=in_filters, nb_filters=nb_filters, init_subsample=init_subsample)
            else:
                layer = basic_block(in_filters=nb_filters, nb_filters=nb_filters, init_subsample=init_subsample)
            layers.append(layer)

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class _upsampling_residual_block(torch.nn.Module):
    def __init__(self, in_filters, nb_filters, repetitions):
        super(_upsampling_residual_block, self).__init__()
        layers = []
        for i in range(repetitions):
            if i == 0: 
                layer = _u_basic_block(in_filters=in_filters, nb_filters=nb_filters)#(input)
            else:
                layer = basic_block(in_filters=nb_filters, nb_filters=nb_filters)#(input)
            layers.append(layer)

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class res_skip(torch.nn.Module):
    def __init__(self):
        super(res_skip, self).__init__()
        self.block0 = _residual_block(in_filters=1, nb_filters=24, repetitions=2, is_first_layer=True)#(input)
        self.block1 = _residual_block(in_filters=24, nb_filters=48, repetitions=3)#(block0)
        self.block2 = _residual_block(in_filters=48, nb_filters=96, repetitions=5)#(block1)
        self.block3 = _residual_block(in_filters=96, nb_filters=192, repetitions=7)#(block2)
        self.block4 = _residual_block(in_filters=192, nb_filters=384, repetitions=12)#(block3)
        
        self.block5 = _upsampling_residual_block(in_filters=384, nb_filters=192, repetitions=7)#(block4)
        self.res1 = _shortcut(in_filters=192, nb_filters=192)#(block3, block5, subsample=(1,1))

        self.block6 = _upsampling_residual_block(in_filters=192, nb_filters=96, repetitions=5)#(res1)
        self.res2 = _shortcut(in_filters=96, nb_filters=96)#(block2, block6, subsample=(1,1))

        self.block7 = _upsampling_residual_block(in_filters=96, nb_filters=48, repetitions=3)#(res2)
        self.res3 = _shortcut(in_filters=48, nb_filters=48)#(block1, block7, subsample=(1,1))

        self.block8 = _upsampling_residual_block(in_filters=48, nb_filters=24, repetitions=2)#(res3)
        self.res4 = _shortcut(in_filters=24, nb_filters=24)#(block0,block8, subsample=(1,1))

        self.block9 = _residual_block(in_filters=24, nb_filters=16, repetitions=2, is_first_layer=True)#(res4)
        self.conv15 = _bn_relu_conv(in_filters=16, nb_filters=1, fh=1, fw=1, subsample=1)#(block7)

    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        x5 = self.block5(x4)
        res1 = self.res1(x3, x5)

        x6 = self.block6(res1)
        res2 = self.res2(x2, x6)

        x7 = self.block7(res2)
        res3 = self.res3(x1, x7)

        x8 = self.block8(res3)
        res4 = self.res4(x0, x8)

        x9 = self.block9(res4)
        y = self.conv15(x9)

        return y
##  end: manga


class PreprocessorLineart(Preprocessor):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.tags = ['Lineart']
        self.model_filename_filters = ['lineart', 'any-test-like']
        # use standard resolution slider
        self.slider_1 = PreprocessorParameter(visible=False)
        self.slider_2 = PreprocessorParameter(visible=False)
        self.sorting_priority = 100

        self.model = None
        self.device = devices.get_device_for('controlnet')

    def load_manga_model(self):
        model_dir = os.path.join(preprocessor_dir, 'manga_line')
        remote_model_path = 'https://huggingface.co/lllyasviel/Annotators/resolve/main/erika.pth'
        model_path = os.path.join(model_dir, 'erika.pth')
        if not os.path.exists(model_path):
            load_file_from_url(remote_model_path, model_dir=model_dir)

        net = res_skip()
        ckpt = torch.load(model_path)
        for key in list(ckpt.keys()):
            if key.startswith('module.'):
                ckpt[key[7:]] = ckpt.pop(key)
        net.load_state_dict(ckpt)
        net.eval()
        self.model = net

    def __call__(self, input_image, resolution, slider_1=None, slider_2=None, slider_3=None, **kwargs):
        image, remove_pad = resize_image_with_pad(input_image, resolution)

        match self.name:
            case 'lineart_anime_inverted':
                if self.model is None:
                    self.load_manga_model()

                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = numpy.ascontiguousarray(image)
                image = torch.from_numpy(image).to(torch.float32).to(self.device)
                image = rearrange(image, 'h w -> 1 1 h w')

                self.model.to(self.device)
                with torch.no_grad():
                    line = self.model(image)[0, 0]
                self.model.cpu()

                line = line.cpu().numpy()
                if 'inverted' in self.name:
                    result = line.clip(0, 255).astype(numpy.uint8)
                else:
                    result = 255 - line.clip(0, 255).astype(numpy.uint8)

            case 'lineart_xDoG' | 'lineart_xDoG_inverted':
                sigma = 1.0
                k = 1.6

                image = image.astype(numpy.float32)
                g0 = cv2.GaussianBlur(image, (3,3), sigma,   borderType=cv2.BORDER_REPLICATE)
                g1 = cv2.GaussianBlur(image, (5,5), sigma*k, borderType=cv2.BORDER_REPLICATE)

                dog = (127.5 + numpy.min(g1-g0, axis=2)).clip(0, 255).astype(numpy.uint8)
                result = numpy.zeros_like(image, dtype=numpy.uint8)
                if 'inverted' in self.name:
                    result[dog < 128] = 255
                else:
                    result[dog >= 128] = 255

            case _:
                return input_image

        return HWC3(remove_pad(result))


add_supported_preprocessor(PreprocessorLineart('lineart_anime_inverted'))
add_supported_preprocessor(PreprocessorLineart('lineart_xDoG'))
add_supported_preprocessor(PreprocessorLineart('lineart_xDoG_inverted'))
