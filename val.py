from ultralytics import YOLO
import logging
from ultralytics.utils import LOGGER

LOGGER.addHandler(logging.FileHandler('results.txt', mode='w'))

model = YOLO('runs/yolo114_VOC/n3/weights/best.pt')
LOGGER.info("Model info:")
LOGGER.info(model)
LOGGER.info(model.info())



metrics = model.val(data='VOC.yaml', batch=64)

LOGGER.info("mAP50-95:" + str(metrics.box.map))  # mAP50-95
LOGGER.info("mAP50: " + str(metrics.box.map50))  # mAP50
LOGGER.info("mAP75: " + str(metrics.box.map75))  # mAP75
LOGGER.info("list of mAP50-95 for each category:" + str(metrics.box.maps))