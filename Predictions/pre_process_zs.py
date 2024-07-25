from torchvision import transforms
from RandAugment import RandAugment

class ResizeImageInterRes():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
      w, h = (img.size[0], img.size[1])
      if (w > h):
          th = 224
          tw = int(round(float(w) / float(h) * 32) / 32.0 * 224)
      else:
          tw = 224
          th = int(round(float(h) / float(w) * 32) / 32.0 * 224)
      return img.resize((tw, th))
      # return img.resize((224, 224))

class ResizeImageInterViT():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
      w, h = (img.size[0], img.size[1])
      if (w > h):
          th = 224
          tw = int(round(float(w) / float(h) * 16) / 16.0 * 224)
      else:
          tw = 224
          th = int(round(float(h) / float(w) * 16) / 16.0 * 224)
      return img.resize((tw, th))
      # return img.resize((224, 224))

class ResizeImage():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
      return img.resize((224, 224))

def image_test():
  normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                   std=[0.26862954, 0.26130258, 0.27577711])
  return transforms.Compose([
    ResizeImage(224),
    transforms.ToTensor(),
    normalize
  ])

def image_test_interRes():
  normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                   std=[0.26862954, 0.26130258, 0.27577711])
  return transforms.Compose([
    ResizeImageInterRes(224),
    transforms.ToTensor(),
    normalize
  ])

def image_test_interViT():
  normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                   std=[0.26862954, 0.26130258, 0.27577711])
  return transforms.Compose([
    ResizeImageInterViT(224),
    transforms.ToTensor(),
    normalize
  ])
