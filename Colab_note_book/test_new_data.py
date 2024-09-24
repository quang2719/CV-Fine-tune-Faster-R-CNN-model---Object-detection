import os
import numpy as np
import cv2
import argparse
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

torch.multiprocessing.set_sharing_strategy('file_system')


def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)


def get_args():
    parser = argparse.ArgumentParser(description="Train faster rcnn model")
    parser.add_argument("--image_path", "-i", type=str, help="path to image", required=True)
    parser.add_argument("--saved_checkpoint", "-o", type=str, default="trained_models/best.pt", help="Load from this checkpoint")
    parser.add_argument("--conf_threshold", "-c", type=float, default=0.3, help="Confident threshold")
    args = parser.parse_args()
    return args

categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonitor']

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_mobilenet_v3_large_320_fpn()
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels, num_classes=21)
    checkpoint = torch.load(args.saved_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.float()
    # model.to(device)
    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1))/255.
    # image = [torch.from_numpy(image).to(device).float()]
    image = [torch.from_numpy(image).float()]
    model.eval()
    with torch.no_grad():
        output = model(image)[0]
        bboxes = output["boxes"]
        labels = output["labels"]
        scores = output["scores"]
        for bbox, label, score in zip(bboxes, labels, scores):
            if score > args.conf_threshold:
                xmin, ymin, xmax, ymax = bbox
                cv2.rectangle(ori_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 3)
                category = categories[label]
                cv2.putText(ori_image, category, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX ,
                            1, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imwrite("prediction.jpg", ori_image)








if __name__ == '__main__':
    args = get_args()
    test(args)