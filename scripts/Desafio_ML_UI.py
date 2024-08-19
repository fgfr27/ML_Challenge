import gradio as gr
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from PIL import Image
import numpy as np

# Configuração do modelo
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85  # Definir threshold para a predição
cfg.MODEL.WEIGHTS = "C:\\Users\\new_d\\OneDrive\\Documents\\output\\model_final.pth"  # Caminho do modelo treinado
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Supondo que treinou com 2 classes: pessoa e gato (além do fundo)

# Mapeamento de IDs para nomes de classes
class_names = ["Person", "Cat"]

predictor = DefaultPredictor(cfg)

# Função de predição
def detect_objects(image):
    image_np = np.array(image)
    outputs = predictor(image_np)
    instances = outputs["instances"].to("cpu")
    
    v = Visualizer(image_np[:, :, ::-1], scale=1.2, instance_mode=ColorMode.IMAGE)
    
    # Adicionar rótulos de classe nas caixas
    v = v.draw_instance_predictions(instances)
    
    result_img = Image.fromarray(v.get_image()[:, :, ::-1])
    
    # Extrair classes detectadas
    detected_classes = instances.pred_classes.numpy()
    
    labels = [class_names[i] for i in detected_classes]
    if not labels:
        labels.append("No cats or people detected")
    
    return result_img, "\n".join(labels)

# Configuração do Gradio
interface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="pil"), gr.Textbox()],
    title="Cat and Person Detector",
    description="Upload an image to see if cats or people are detected."
)

# Iniciar a interface
interface.launch()
