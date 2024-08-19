from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
import os

def register_coco_instances(name, json_file, image_root):
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file, image_root=image_root)

dataset_dir = "D:\\Felipe"
train_images = os.path.join(dataset_dir, "train2017_converted")
val_images = os.path.join(dataset_dir, "val2017_converted")
train_ann = os.path.join(dataset_dir, "annotations/instances_train2017.json")
val_ann = os.path.join(dataset_dir, "annotations/instances_val2017.json")

register_coco_instances("coco_train_person_cat", train_ann, train_images)
register_coco_instances("coco_val_person_cat", val_ann, val_images)
print("Foi concluido o dataset_register")
