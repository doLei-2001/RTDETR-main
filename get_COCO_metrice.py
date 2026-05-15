import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tidecv import TIDE, datasets

# COCO指标如果一直生成不出来之类的问题可以看这期视频排查：https://www.bilibili.com/video/BV1SdNizEE4X/
# visdrone数据集：python get_COCO_metrice.py --pred_json /home/public/leiao23/code/RTDETR-main/runs/val/exp_rtdetr_r18_150epoch/predictions.json --anno_json /home/public/leiao23/code/datasets/dataset_visdrone/VisDrone2019-DET-test-dev/test.json
# bdd100k数据集：python get_COCO_metrice.py --pred_json /home/public/leiao23/code/RTDETR-main/runs/val/exp-rtdetr-bdd100k-300epoch-r18-/predictions.json  --anno_json /home/public/leiao23/code/RTDETR-main/dataset/bdd100k/data.json

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_json', type=str, default='data.json', help='training model path')
    parser.add_argument('--pred_json', type=str, default='', help='data yaml path')
    
    return parser.parse_known_args()[0]

if __name__ == '__main__':
    opt = parse_opt()
    anno_json = opt.anno_json
    pred_json = opt.pred_json
    
    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    
    tide = TIDE()
    tide.evaluate_range(datasets.COCO(anno_json), datasets.COCOResult(pred_json), mode=TIDE.BOX)
    tide.summarize()
    tide.plot(out_dir='result')