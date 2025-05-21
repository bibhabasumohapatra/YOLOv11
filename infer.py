import os
import json
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

# Replace this with your actual model
from model import DetectionModel
from utils import non_max_suppression

# Configuration
COCO_PATH = '/home/bibhabasum/projects/IIIT/data/coco/'
IMG_DIR = os.path.join(COCO_PATH, 'val2017')
ANN_FILE = os.path.join(COCO_PATH, 'annotations/instances_val2017.json')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONF_THRESH = 0.8  # Lower to let NMS do filtering
IOU_THRESH = 0.65

# Transform
transform = T.Compose([
    T.Resize((640, 640)),  # Resize as per your model input
    T.ToTensor(),
])

# Load model

model = DetectionModel(nc = 80)

    # print(model)
weights = torch.load('/home/bibhabasum/projects/IIIT/ultralytics/model_state_dict.pth',map_location="cpu") # Load weights
model.load_state_dict(weights, strict=True)  # Load model weights
model.to(DEVICE)
model.eval()

# Load COCO
coco = COCO(ANN_FILE)
img_ids = coco.getImgIds()

results = []
n = []
for img_id in tqdm(img_ids):
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(IMG_DIR, img_info['file_name'])
    img = Image.open(img_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = model(input_tensor) # Get predictions in [cx, cy, w, h, conf, cls...]

    # Apply your custom NMS
    k = 489714
    detections = non_max_suppression(preds, CONF_THRESH, IOU_THRESH)[0] ## as one image only take one input

    # Convert to COCO format
    for det in detections:
        x1, y1, width, height, score, cls_id = det
        # width = x2 - x1
        # width = y2 - y1
        # Map class ID to category ID using classes_dict
        from classes_dict import classes_dict
        class_name = classes_dict[int(cls_id)]
        category_id = coco.getCatIds(catNms=[class_name])[0]
        
        if class_name == "zebra":
            print(category_id, cls_id)

        results.append({
            "image_id": img_id,
            "category_id": category_id,
            "bbox": [x1.item(), y1.item(), width.item(), height.item()],
            "score": score.item(),
        })
    print(img_id, results)
    break
# print(set(n))
# Save results
with open('coco_results.json', 'w') as f:
    json.dump(results, f)

# Evaluate
coco_dt = coco.loadRes('coco_results.json')
coco_eval = COCOeval(coco, coco_dt, iouType='bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
