from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random

from torchvision import transforms

""" 1. Graduatally increase magnitude 
    2. Increase number of operationis in one policy
"""



class PolicySeq(object):
    """magnitude increase when each subpolicy has two operations  
    """
    def __init__(self, mag_idx=4, prob=0.5, n_op=2, fillcolor=(128, 128, 128)):
        """

        Args:
        ----------
            mag idx: 
            fillcolor (tuple, optional):  Defaults to (128, 128, 128).
        """
        self.op_pool = ["autocontrast", "equalize", "invert"]
        #self.op_pool = ["posterize",  "sharpness", "contrast", "brightness", "color", "solarize"]   # "posterize",  "sharpness", "contrast", "brightness", "color", solarize
        #self.op_pool = ["autocontrast", "equalize", "invert", "posterize",  "sharpness", "contrast", "brightness", "color", "solarize"]

        self.prob = prob
        self.n_op = n_op
        self.magnitude_idx = mag_idx
        self.fillcolor = fillcolor

    def __call__(self, img):
        op_seq = []
        index_seq = []
        # repeat allow
        # for _ in range(self.n_op):
        #     policy_idx = random.randint(0, len(self.op_pool) - 1)
        #     op_seq.append((self.prob, self.op_pool[policy_idx], self.magnitude_idx))
        
        # repeat not allow
        op_choices = random.sample(self.op_pool, self.n_op)
        for op_name in op_choices:
            op_seq.append((self.prob, op_name, self.magnitude_idx))

        policy = SubPolicySeq(op_seq, self.fillcolor)

        return policy(img)

    def __repr__(self):
        return "Augment Magnitude Increase Policy"


class PolicySeqBlackBG(object):
    """magnitude increase when each subpolicy has two operations  
    """
    def __init__(self, mag_idx=4, prob=0.5, n_op=2, fillcolor=(128, 128, 128)):
        """

        Args:
        ----------
            mag idx: 
            fillcolor (tuple, optional):  Defaults to (128, 128, 128).
        """
        self.op_pool = ["autocontrast", "equalize", "invert"]
        #self.op_pool = ["posterize",  "sharpness", "contrast", "brightness", "color", "solarize"]   # "posterize",  "sharpness", "contrast", "brightness", "color", solarize
        #self.op_pool = ["autocontrast", "equalize", "invert", "posterize",  "sharpness", "contrast", "brightness", "color", "solarize"]

        self.prob = prob
        self.n_op = n_op
        self.magnitude_idx = mag_idx
        self.fillcolor = fillcolor
        self.toPIL = transforms.ToPILImage()
        self.transform = transforms.Compose([
            transforms.CenterCrop(127),
            transforms.Resize(size=224, interpolation=2),
            transforms.ToTensor()
        ])

    def __call__(self, img):
        img_origin_arr = np.asarray(img)
        img = img.convert('RGB')
        op_seq = []

        # repeat not allow
        op_choices = random.sample(self.op_pool, self.n_op)
        for op_name in op_choices:
            op_seq.append((self.prob, op_name, self.magnitude_idx))

        policy = SubPolicySeq(op_seq, self.fillcolor)
        img_aug = policy(img)
        img_aug_arr = np.asarray(img_aug)
        
        img_aug_arr = img_aug_arr.copy()
        img_aug_arr[img_origin_arr[:, :, 3] == 0] = [0, 0, 0]
        img_aug = self.toPIL(img_aug_arr)
        img_aug = self.transform(img_aug)

        return img_aug

    def __repr__(self):
        return "Augment Magnitude Increase Policy"


class GeometricPolicySeq(object):
    """magnitude increase when each subpolicy has two operations  
    """
    def __init__(self, n_op=2, rotate_degree=30, out_size=127, fillcolor=(128, 128, 128)):
        """

        Args:
        ----------
            mag idx: 
            fillcolor (tuple, optional):  Defaults to (128, 128, 128).
        """
        self.op_pool = [transforms.RandomCrop(out_size), 
                transforms.RandomHorizontalFlip(),  
                transforms.RandomRotation((-1 * rotate_degree, rotate_degree))] #  
                            
        self.n_op = n_op
        self.fillcolor = fillcolor
        self.out_size = out_size
        self.center_crop = transforms.CenterCrop(out_size)

    def __call__(self, img):
        
        # repeat not allow
        op_choices = random.sample(self.op_pool, self.n_op)
        for op in op_choices:
            img = op(img)
            
        if img.size[0] > self.out_size:
            img = self.center_crop(img)

        return img

    def __repr__(self):
        return "Geometric Increase Policy"

class GeometricPolicyMag(object):
    """magnitude increase when each subpolicy has two operations  
    """
    def __init__(self, mag_idx=0, max_angle=60, max_trans_ratio=0.3, level=3, out_size=127, fillcolor=(128, 128, 128)):
        """

        Args:
        ----------
            mag idx: 
            fillcolor (tuple, optional):  Defaults to (128, 128, 128).
        """
        
        self.mag_idx = mag_idx
        degree = (max_angle / level) * mag_idx
        trans_ratio = (max_trans_ratio / level) * mag_idx
        print(f"Random Geometric Augmentation, Leval:{mag_idx}, Rot:{degree}, Trans:{trans_ratio}")
        self.transform = transforms.RandomAffine(degrees=degree, translate=(trans_ratio, trans_ratio))
        self.fillcolor = fillcolor
        
        self.center_crop = transforms.CenterCrop(out_size)
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = self.transform(img)
        img = self.center_crop(img)

        return img

    def __repr__(self):
        return "Geometric Increase Policy"


class ColorAugFore(object):
    """

    Args:
        object ([type]): [description]

    Returns:
        [type]: [description]
    """
    def __init__(self, magnitude):
        self.magnitude = magnitude     # 0 10 20 30   
        self.toPIL = transforms.ToPILImage()
    def __call__(self, img):
        img_raw = np.array(img)
        img_arr = img_raw[:, :, :3]
        r_aug = 2 * self.magnitude * np.random.rand() - self.magnitude
        g_aug = 2 * self.magnitude * np.random.rand() - self.magnitude
        b_aug = 2 * self.magnitude * np.random.rand() - self.magnitude
        img_arr[img_raw[:, :, 3] != 0, 0] = img_arr[img_raw[:, :, 3] != 0, 0] + r_aug
        img_arr[img_raw[:, :, 3] != 0, 1] = img_arr[img_raw[:, :, 3] != 0, 1] + g_aug
        img_arr[img_raw[:, :, 3] != 0, 2] = img_arr[img_raw[:, :, 3] != 0, 2] + b_aug
        
        img_arr = np.clip(img_arr, 0, 255)
        img = self.toPIL(img_arr)

        return img

    def __repr__(self):
        return super().__repr__()


class SubPolicySeq(object):
    def __init__(self, op_seq, fillcolor=(128, 128, 128)):
        self.ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        self.func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }
        self.op_seq = op_seq

    def __call__(self, img):

        for op in self.op_seq:
            p, op_name, magnitude_idx = op
            operation = self.func[op_name]
            magnitude = self.ranges[op_name][magnitude_idx]
            if random.random() < p:
                #print(op_name, magnitude, "Reported")
                img = operation(img, magnitude)
        return img 

    def __repr__(self):
        rep = ""
        for op in self.op_seq:
            rep += f"({op[0]} {op[1]} {op[2]})   "
        
        return rep

    

class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]


    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img


if __name__ == "__main__":
    pass
