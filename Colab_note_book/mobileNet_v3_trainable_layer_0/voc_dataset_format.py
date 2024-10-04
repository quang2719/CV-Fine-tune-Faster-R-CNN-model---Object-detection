from torchvision.datasets import VOCDetection
from torchvision.transforms import Compose, ToPILImage,ToTensor
from pprint import pprint
import matplotlib.pyplot as plt
import torch


class VOCDataset(VOCDetection):
  def __init__(self,root,year,image_set,download,transform):
    super().__init__(root, year,image_set,download,transform)
    self.categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 
              'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 
              'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

  def __getitem__(self, item):
    image,data = super().__getitem__(item)
    bboxes = []
    labels = []
    for obj in data['annotation']['object']:
      bboxes.append([
        int(obj['bndbox']['xmin']),
        int(obj['bndbox']['ymin']),
        int(obj['bndbox']['xmax']),
        int(obj['bndbox']['ymax']),

      ])
      labels.append(self.categories.index(obj['name']))
    bboxes = torch.FloatTensor(bboxes)
    labels = torch.LongTensor(labels)

    target = {
      'boxes': bboxes,
      'labels': labels
    }

    return image,target

if __name__ == '__main__':
  transform = ToTensor()
  dataset = VOCDataset(
      root = '/content/drive/MyDrive/Faster_R_CNN_project/Dataset',
      year = '2012',
      image_set = 'train',
      download = False,
      transform = transform
    )
  image,target = dataset[1500]
  pprint(type(image))
  pprint(image.shape)
  pprint(target)
  pprint(image)




  