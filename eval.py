import os
import time
import copy
import json
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.log as logger
from lib import RPC, S2MC2Eval
from lib import convert_eval_format, post_process, merge_outputs
from lib import visual_image
from lib.model_utils.config import config, dataset_config, net_config, eval_config
from lib.model_utils.device_adapter import get_device_id

_current_dir = os.path.dirname(os.path.realpath(__file__))


def predict():
    '''
    Predict function
    '''
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    enable_nms_fp16 = True

    logger.info("Begin creating {} dataset".format(config.run_mode))
    rpc = RPC(dataset_config, run_mode=config.run_mode, net_opt=net_config,
               enable_visual_image=config.visual_image, save_path=config.save_result_dir, )
    rpc.init(config.data_dir, keep_res=eval_config.keep_res)
    dataset = rpc.create_eval_dataset()

    net_for_eval = S2MC2Eval(net_config, eval_config.K, enable_nms_fp16)
    net_for_eval.set_train(False)

    param_dict = load_checkpoint(config.load_checkpoint_path)
    load_param_into_net(net_for_eval, param_dict)

    # save results
    save_path = os.path.join(config.save_result_dir, config.run_mode)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if config.visual_image == "true":
        save_pred_image_path = os.path.join(save_path, "pred_image")
        if not os.path.exists(save_pred_image_path):
            os.makedirs(save_pred_image_path)
        save_gt_image_path = os.path.join(save_path, "gt_image")
        if not os.path.exists(save_gt_image_path):
            os.makedirs(save_gt_image_path)

    total_nums = dataset.get_dataset_size()
    print("\n========================================\n")
    print("Total images num: ", total_nums)
    print("Processing, please wait a moment.")

    pred_annos = {"images": [], "annotations": []}

    index = 0
    for data in dataset.create_dict_iterator(num_epochs=1):
        index += 1
        image = data['image']
        image_id = data['image_id'].asnumpy().reshape((-1))[0]

        # run prediction
        start = time.time()
        detections = []
        for scale in eval_config.multi_scales:
            images, meta = rpc.pre_process_for_test(image.asnumpy(), image_id, scale)
            detection = net_for_eval(Tensor(images))
            dets = post_process(detection.asnumpy(), meta, scale, dataset_config.num_classes)
            detections.append(dets)
        end = time.time()
        print("Image {}/{} id: {} cost time {} ms".format(index, total_nums, image_id, (end - start) * 1000.))

        # post-process
        detections = merge_outputs(detections, dataset_config.num_classes, eval_config.SOFT_NMS)
        # get prediction result
        pred_json = convert_eval_format(detections, image_id, eval_config.valid_ids)
        gt_image_info = rpc.coco.loadImgs([image_id])

        for image_info in pred_json["images"]:
            pred_annos["images"].append(image_info)
        for image_anno in pred_json["annotations"]:
            pred_annos["annotations"].append(image_anno)
        if config.visual_image == "true":
            img_file = os.path.join(rpc.image_path, gt_image_info[0]['file_name'])
            gt_image = cv2.imread(img_file)
            if config.run_mode != "test":
                annos = rpc.coco.loadAnns(rpc.anns[image_id])
                visual_image(copy.deepcopy(gt_image), annos, save_gt_image_path,
                             score_threshold=eval_config.score_thresh)
            anno = copy.deepcopy(pred_json["annotations"])
            visual_image(gt_image, anno, save_pred_image_path, score_threshold=eval_config.score_thresh)

    # save results
    save_path = os.path.join(config.save_result_dir, config.run_mode)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pred_anno_file = os.path.join(save_path, '{}_pred_result.json').format(config.run_mode)
    json.dump(pred_annos, open(pred_anno_file, 'w'))
    pred_res_file = os.path.join(save_path, '{}_pred_eval.json').format(config.run_mode)
    json.dump(pred_annos["annotations"], open(pred_res_file, 'w'))

    if config.run_mode != "test" and config.enable_eval:
        run_eval(rpc.annot_path, pred_res_file)


def run_eval(gt_anno, pred_anno):
    """evaluation by coco api"""
    rpc = COCO(gt_anno)
    rpc_dets = rpc.loadRes(pred_anno)
    rpc_eval = COCOeval(rpc, rpc_dets, "bbox")
    rpc_eval.evaluate()
    rpc_eval.accumulate()
    rpc_eval.summarize()


if __name__ == "__main__":
    predict()
