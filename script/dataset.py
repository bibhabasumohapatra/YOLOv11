import os
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class COCODataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.img_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract bounding boxes and class IDs
        bboxes = []
        class_ids = []
        for ann in anns:
            bbox = ann['bbox']  # [x, y, width, height]
            category_id = ann['category_id']
            # Convert bbox to [x_min, y_min, x_max, y_max] format
            x_min, y_min, w, h = bbox
            x_max = x_min + w
            y_max = y_min + h
            bboxes.append([x_min, y_min, x_max, y_max])
            class_ids.append(category_id)

        # Normalize bounding boxes
        h, w, _ = image.shape
        bboxes = [[x_min / w, y_min / h, x_max / w, y_max / h] for x_min, y_min, x_max, y_max in bboxes]

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_ids)
            image = transformed['image']
            bboxes = transformed['bboxes']
            class_ids = transformed['class_labels']

        # Convert to tensors
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        class_ids = torch.tensor(class_ids, dtype=torch.int64)

        return image, bboxes, class_ids


# Example usage
if __name__ == "__main__":
    transform = A.Compose([
        A.Resize(640, 640, always_apply=True),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco',  
                                label_fields=['class_labels']))

    dataset = COCODataset(
        img_dir='/home/bibhabasum/projects/IIIT/data/coco/val2017',
        ann_file='/home/bibhabasum/projects/IIIT/data/coco/annotations/instances_val2017.json',

        transform=transform,
    )
    def collate_fn(batch):
        images, bboxes, class_ids = zip(*batch)
        max_bboxes = max([bbox.shape[0] for bbox in bboxes])
        padded_bboxes = [torch.cat([bbox, torch.zeros(max_bboxes - bbox.shape[0], 4)], dim=0) for bbox in bboxes]
        padded_class_ids = [torch.cat([cls_id, torch.zeros(max_bboxes - cls_id.shape[0], dtype=torch.int64)]) for cls_id in class_ids]
        return torch.stack(images), torch.stack(padded_bboxes), torch.stack(padded_class_ids)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    for images, bboxes, class_ids in dataloader:
        print(images.shape)
        print(bboxes.shape)
        print(class_ids.shape)
        break