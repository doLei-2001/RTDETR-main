import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
from ultralytics import YOLO

# 预测框粗细和颜色修改问题可看<使用说明.md>下方的<YOLOV8源码常见疑问解答小课堂>第六点

if __name__ == '__main__':
    # model = RTDETR('/home/public/leiao23/code/RTDETR-main/weights/rtdetr-r18.pt') # select your model.pt path
    model = RTDETR('/home/public/leiao23/code/RTDETR-main/runs/train/exp-rtdetr-bdd100k-300epoch-rtdetr-SRFD-SOEP-MFM-Pola-SEFN-Mona-DIMB/weights/best.pt') # select your model.pt path
    # model = YOLO('/home/public/wuan23/yolo11n/run_bddk/yolo11_bddk/weights/best.pt')
    model.predict(
          # source='/home/public/leiao23/code/datasets/dataset_visdrone/VisDrone2019-DET-val/images',
          source='/home/public/wuan23/data/BDD100K/test/images',
          conf=0.6,
          project='runs/detect',
          name='new-exp-rtdetr-bdd100k-dimb-test-conf0.6',
          save=True,
          # visualize=True # visualize model features maps
          line_width=2, # line width of the bounding boxes
          show_conf=True, # do not show prediction confidence
          # show_labels=False, # do not show prediction labels
          save_txt=True, # save results as .txt file
          # save_crop=True, # save cropped images with results
          )