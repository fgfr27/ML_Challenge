from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer

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
cfg.OUTPUT_DIR = "./output2"

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
print('Modelo configurado..')