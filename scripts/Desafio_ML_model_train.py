from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import build_detection_train_loader, DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from Desafio_ML_dataset_mapper import NPYDatasetMapper
from detectron2.checkpoint import Checkpointer
from detectron2.checkpoint import DetectionCheckpointer

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

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Ajuste para o número total de classes
cfg.DATASETS.TRAIN = ("coco_train_person_cat",)
cfg.DATASETS.TEST = ("coco_val_person_cat",)
cfg.DATALOADER.NUM_WORKERS = 4

# IDs para "cat" e "person"
cfg.DATASETS.CAT_ID = 17  # Substitua pelo ID correto no seu dataset
cfg.DATASETS.PERSON_ID = 1  # Substitua pelo ID correto no seu dataset

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0001
cfg.SOLVER.MAX_ITER = 5000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.MASK_ON = False
cfg.OUTPUT_DIR = "./output"

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
print('Modelo configurado..')

class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=NPYDatasetMapper(cfg, is_train=True))

# Bloco principal
if __name__ == '__main__':
    
    trainer = Trainer(cfg)
    # Checkpointer para carregar pesos sem as camadas da cabeça de ROI
    DetectionCheckpointer(trainer.model).load(cfg.MODEL.WEIGHTS)
    # Adicionar o Checkpointer para salvar o modelo
    checkpointer = Checkpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
    trainer.resume_or_load(resume=False)
    trainer.train()
    print("Modelo treinado..")
    # Salvar o modelo treinado
    checkpointer.save("model_final")
    print("Modelo salvo..")