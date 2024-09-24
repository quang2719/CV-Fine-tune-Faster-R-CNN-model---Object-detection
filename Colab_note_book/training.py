import os
import numpy as np
from voc_dataset_format import VOCDataset
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, RandomAffine, ColorJitter
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

torch.multiprocessing.set_sharing_strategy('file_system')

# Function to get arguments
def get_arg():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN')
    parser.add_argument('--data_path', '-d', type=str, default='/content/drive/MyDrive/Faster_R_CNN_project/Dataset', help='Path to dataset')
    parser.add_argument("--year", "-y", type=str, default="2012")
    parser.add_argument('--num_epochs', '-n', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', '-l', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum')
    parser.add_argument('--log_folder', '-p', type=str, default='tensorboard/pascal_voc', help='Path to generate log folder')
    parser.add_argument('--checkpoint_folder', '-c', type=str, default='checkpoint', help='Path to generate checkpoint folder')
    parser.add_argument("--saved_checkpoint", "-o", type=str, default=None, help="Continue from this checkpoint")
    args = parser.parse_args()
    return args

def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)

def train_data(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define transformations
    train_transform = Compose([
        RandomAffine( 
          # Data augumentation
          # Chane location
            degrees=(-5, 5), 
            translate=(0.15, 0.15),
            scale=(0.85, 1.15),
            shear=10
        ),
        ColorJitter(
          # Data augumentation
          # Chane color
            brightness=0.125,
            contrast=0.5,
            saturation=0.5,
            hue=0.05
        ),
        ToTensor(),  # No need to Normalize
    ])
    
    # Load datasets
    train_dataset = VOCDataset(
        root=args.data_path,
        year=args.year,
        image_set="train",
        download=False,
        transform=train_transform
    )
    val_dataset = VOCDataset(
        root=args.data_path,
        year=args.year,
        image_set="val",
        download=False,
        transform=ToTensor()
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Load model
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    ).to(device)
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels,
                                                      num_classes=len(train_dataset.categories))
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=args.log_folder)
    best_map = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        for iter, (images, labels) in enumerate(train_dataloader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in labels]

            optimizer.zero_grad()
            with autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            if iter % 10 == 0:
                print(f"Epoch [{epoch}/{args.num_epochs}], Iter [{iter}/{len(train_dataloader)}], Loss: {losses.item()}")

        # Validation
        model.eval()
        metric = MeanAveragePrecision(iou_type="bbox")
        for iter, (images, labels) in enumerate(val_dataloader):
            images = [image.to(device) for image in images]
            with torch.no_grad():
                outputs = model(images)
            preds = []

            for output in outputs:
                preds.append({
                    "boxes": output["boxes"].to("cpu"),
                    "scores": output["scores"].to("cpu"),
                    "labels": output["labels"].to("cpu"),
                })

            targets = []
            for label in labels:
                targets.append({
                    "boxes": label["boxes"],
                    "labels": label["labels"],
                })
            metric.update(preds, targets)
        
        result = metric.compute()
        pprint(result)
        writer.add_scalar("Val/mAP", result["map"], epoch)
        writer.add_scalar("Val/mAP_50", result["map_50"], epoch)
        writer.add_scalar("Val/mAP_75", result["map_75"], epoch)

        checkpoint = {
            "model_state_dict": model.state_dict(), #store parameter
            "map": result["map"],
            "epoch": epoch + 1,
            "optimizer_state_dict": optimizer.state_dict()
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_folder, "last.pt")) # save parameter
        if result["map"] > best_map: #store version with best mAP 
            best_map = result["map"]
            torch.save(checkpoint, os.path.join(args.checkpoint_folder, "best.pt"))

if __name__ == '__main__':
    args = get_arg()
    train_data(args)