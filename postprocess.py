import os
import json
import numpy as np
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
from lib.model_utils.config import config, dataset_config, eval_config
from lib import convert_eval_format, post_process, merge_outputs


def cal_acc(result_path, label_path, meta_path, save_path):
    """calculate inference accuracy"""
    name_list = np.load(os.path.join(meta_path, "name_list.npy"), allow_pickle=True)
    meta_list = np.load(os.path.join(meta_path, "meta_list.npy"), allow_pickle=True)

    label_infor = coco.COCO(label_path)
    pred_annos = {"images": [], "annotations": []}
    for num, image_id in enumerate(name_list):
        meta = meta_list[num]
        pre_image = np.fromfile(os.path.join(result_path) + "/eval2017_image_" + str(image_id) + "_0.bin",
                                dtype=np.float32).reshape((1, 100, 6))
        detections = []
        for scale in eval_config.multi_scales:
            dets = post_process(pre_image, meta, scale, dataset_config.num_classes)
            detections.append(dets)
        detections = merge_outputs(detections, dataset_config.num_classes, eval_config.SOFT_NMS)
        pred_json = convert_eval_format(detections, image_id, eval_config.valid_ids)
        label_infor.loadImgs([image_id])
        for image_info in pred_json["images"]:
            pred_annos["images"].append(image_info)
        for image_anno in pred_json["annotations"]:
            pred_annos["annotations"].append(image_anno)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pred_anno_file = os.path.join(save_path, '{}_pred_result.json').format(config.run_mode)
    json.dump(pred_annos, open(pred_anno_file, 'w'))
    pred_res_file = os.path.join(save_path, '{}_pred_eval.json').format(config.run_mode)
    json.dump(pred_annos["annotations"], open(pred_res_file, 'w'))

    rpc_anno = coco.COCO(label_path)
    rpc_dets = rpc_anno.loadRes(pred_res_file)
    rpc_eval = COCOeval(rpc_anno, rpc_dets, "bbox")
    rpc_eval.evaluate()
    rpc_eval.accumulate()
    rpc_eval.summarize()


if __name__ == '__main__':
    cal_acc(config.result_path, config.label_path, config.meta_path, config.save_path)
