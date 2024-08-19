import os
import numpy as np
import torch
import copy
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import Boxes, Instances

class NPYDatasetMapper:
    def __init__(self, cfg, is_train=True):
        super(NPYDatasetMapper, self).__init__()
        self.cfg = cfg  # Salvar cfg como um atributo da classe
        self.augmentations = T.AugmentationList([
            T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, sample_style='choice'),
            T.RandomFlip(),
        ]) if is_train else T.AugmentationList([
            T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, sample_style='choice')
        ])
        self.is_train = is_train

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # Faz uma cópia dos dados

        # Ajuste o caminho da imagem para .npy
        image_path = dataset_dict["file_name"].replace('.jpg', '.npy')
        
        try:
            image = np.load(image_path)  # Carrega a imagem .npy
            if image.size == 0:
                raise ValueError(f"Array de imagem vazio para o caminho: {image_path}")
        except FileNotFoundError as e:
            print(f"Arquivo não encontrado: {image_path}")
            return None  # Ignora amostras onde o arquivo não é encontrado
        
        # Verifica se há valores NaN ou infinitos e substitui por zero
        if np.isnan(image).any() or np.isinf(image).any():
            print(f"Imagem com valores inválidos: {image_path}")
            image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Converte a imagem para um tensor
        image = torch.tensor(image).permute(2, 0, 1)  # Muda a ordem para CxHxW

        # Adiciona a imagem ao dicionário
        dataset_dict["image"] = image

        if "annotations" in dataset_dict:
            annos = dataset_dict.pop("annotations")
            valid_classes = [0, 1]  # IDs para 'person' e 'cat'
            valid_annos = [anno for anno in annos if anno["category_id"] in valid_classes]
            
            if len(valid_annos) == 0:
                return None  # Ignora amostras sem anotações válidas

            # Cria as instâncias do objeto detectron2
            instances = utils.annotations_to_instances(valid_annos, image.shape[1:], mask_format="bitmask")
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict