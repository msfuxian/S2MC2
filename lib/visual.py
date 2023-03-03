import os
import json
import random
import cv2
import numpy as np
import pycocotools.coco as COCO
from .model_utils.config import eval_config as eval_cfg
from .image import get_affine_transform, affine_transform


rpc_class_name2id = {'1_puffed_food': 1, '2_puffed_food': 2, '3_puffed_food': 3, '4_puffed_food': 4, '5_puffed_food': 5,
                     '6_puffed_food': 6, '7_puffed_food': 7, '8_puffed_food': 8, '9_puffed_food': 9, '10_puffed_food': 10,
                     '11_puffed_food': 11, '12_puffed_food': 12, '13_dried_fruit': 13, '14_dried_fruit': 14,
                     '15_dried_fruit': 15, '16_dried_fruit': 16, '17_dried_fruit': 17, '18_dried_fruit': 18,
                     '19_dried_fruit': 19, '20_dried_fruit': 20, '21_dried_fruit': 21, '22_dried_food': 22,
                     '23_dried_food': 23, '24_dried_food': 24, '25_dried_food': 25, '26_dried_food': 26,
                     '27_dried_food': 27, '28_dried_food': 28, '29_dried_food': 29, '30_dried_food': 30,
                     '31_instant_drink': 31, '32_instant_drink': 32, '33_instant_drink': 33, '34_instant_drink': 34,
                     '35_instant_drink': 35, '36_instant_drink': 36, '37_instant_drink': 37, '38_instant_drink': 38,
                     '39_instant_drink': 39, '40_instant_drink': 40, '41_instant_drink': 41, '42_instant_noodles': 42,
                     '43_instant_noodles': 43, '44_instant_noodles': 44, '45_instant_noodles': 45,
                     '46_instant_noodles': 46, '47_instant_noodles': 47, '48_instant_noodles': 48,
                     '49_instant_noodles': 49, '50_instant_noodles': 50, '51_instant_noodles': 51,
                     '52_instant_noodles': 52, '53_instant_noodles': 53, '54_dessert': 54, '55_dessert': 55,
                     '56_dessert': 56, '57_dessert': 57, '58_dessert': 58, '59_dessert': 59, '60_dessert': 60,
                     '61_dessert': 61, '62_dessert': 62, '63_dessert': 63, '64_dessert': 64, '65_dessert': 65,
                     '66_dessert': 66, '67_dessert': 67, '68_dessert': 68, '69_dessert': 69, '70_dessert': 70,
                     '71_drink': 71, '72_drink': 72, '73_drink': 73, '74_drink': 74, '75_drink': 75, '76_drink': 76,
                     '77_drink': 77, '78_drink': 78, '79_alcohol': 79, '80_alcohol': 80, '81_drink': 81, '82_drink': 82,
                     '83_drink': 83, '84_drink': 84, '85_drink': 85, '86_drink': 86, '87_drink': 87, '88_alcohol': 88,
                     '89_alcohol': 89, '90_alcohol': 90, '91_alcohol': 91, '92_alcohol': 92, '93_alcohol': 93,
                     '94_alcohol': 94, '95_alcohol': 95, '96_alcohol': 96, '97_milk': 97, '98_milk': 98, '99_milk': 99,
                     '100_milk': 100, '101_milk': 101, '102_milk': 102, '103_milk': 103, '104_milk': 104,
                     '105_milk': 105, '106_milk': 106, '107_milk': 107, '108_canned_food': 108, '109_canned_food': 109,
                     '110_canned_food': 110, '111_canned_food': 111, '112_canned_food': 112, '113_canned_food': 113,
                     '114_canned_food': 114, '115_canned_food': 115, '116_canned_food': 116, '117_canned_food': 117,
                     '118_canned_food': 118, '119_canned_food': 119, '120_canned_food': 120, '121_canned_food': 121,
                     '122_chocolate': 122, '123_chocolate': 123, '124_chocolate': 124, '125_chocolate': 125,
                     '126_chocolate': 126, '127_chocolate': 127, '128_chocolate': 128, '129_chocolate': 129,
                     '130_chocolate': 130, '131_chocolate': 131, '132_chocolate': 132, '133_chocolate': 133,
                     '134_gum': 134, '135_gum': 135, '136_gum': 136, '137_gum': 137, '138_gum': 138, '139_gum': 139,
                     '140_gum': 140, '141_gum': 141, '142_candy': 142, '143_candy': 143, '144_candy': 144,
                     '145_candy': 145, '146_candy': 146, '147_candy': 147, '148_candy': 148, '149_candy': 149,
                     '150_candy': 150, '151_candy': 151, '152_seasoner': 152, '153_seasoner': 153, '154_seasoner': 154,
                     '155_seasoner': 155, '156_seasoner': 156, '157_seasoner': 157, '158_seasoner': 158,
                     '159_seasoner': 159, '160_seasoner': 160, '161_seasoner': 161, '162_seasoner': 162,
                     '163_seasoner': 163, '164_personal_hygiene': 164, '165_personal_hygiene': 165,
                     '166_personal_hygiene': 166, '167_personal_hygiene': 167, '168_personal_hygiene': 168,
                     '169_personal_hygiene': 169, '170_personal_hygiene': 170, '171_personal_hygiene': 171,
                     '172_personal_hygiene': 172, '173_personal_hygiene': 173, '174_tissue': 174, '175_tissue': 175,
                     '176_tissue': 176, '177_tissue': 177, '178_tissue': 178, '179_tissue': 179, '180_tissue': 180,
                     '181_tissue': 181, '182_tissue': 182, '183_tissue': 183, '184_tissue': 184, '185_tissue': 185,
                     '186_tissue': 186, '187_tissue': 187, '188_tissue': 188, '189_tissue': 189, '190_tissue': 190,
                     '191_tissue': 191, '192_tissue': 192, '193_tissue': 193, '194_stationery': 194,
                     '195_stationery': 195, '196_stationery': 196, '197_stationery': 197, '198_stationery': 198,
                     '199_stationery': 199, '200_stationery': 200}


def rpc_box_to_bbox(box):
    """convert height/width to position coordinates"""
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
    return bbox


def resize_image(image, anns, width, height):
    """resize image to specified scale"""
    h, w = image.shape[0], image.shape[1]
    c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
    s = max(image.shape[0], image.shape[1]) * 1.0
    trans_output = get_affine_transform(c, s, 0, [width, height])
    out_img = cv2.warpAffine(image, trans_output, (width, height), flags=cv2.INTER_LINEAR)

    num_objects = len(anns)
    resize_anno = []
    for i in range(num_objects):
        ann = anns[i]
        bbox = rpc_box_to_bbox(ann['bbox'])
        bbox[:2] = affine_transform(bbox[:2], trans_output)
        bbox[2:] = affine_transform(bbox[2:], trans_output)
        bbox[0::2] = np.clip(bbox[0::2], 0, width - 1)
        bbox[1::2] = np.clip(bbox[1::2], 0, height - 1)
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if (h > 0 and w > 0):
            ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            bbox = [ct[0] - w / 2, ct[1] - h / 2, w, h, 1]
            ann["bbox"] = bbox
            gt = ann
            resize_anno.append(gt)
    return out_img, resize_anno


def merge_pred(ann_path, mode="val", name="merged_annotations"):
    """merge annotation info of each image together"""
    files = os.listdir(ann_path)
    data_files = []
    for file_name in files:
        if "json" in file_name:
            data_files.append(os.path.join(ann_path, file_name))
    pred = {"images": [], "annotations": []}
    for file in data_files:
        anno = json.load(open(file, 'r'))
        if "images" in anno:
            for img in anno["images"]:
                pred["images"].append(img)
        if "annotations" in anno:
            for ann in anno["annotations"]:
                pred["annotations"].append(ann)
    json.dump(pred, open('{}/{}_{}.json'.format(ann_path, name, mode), 'w'))


def visual(ann_path, image_path, save_path, ratio=1, mode="val", name="merged_annotations"):
    """visulize all images based on dataset and annotations info"""
    merge_pred(ann_path, mode, name)
    ann_path = os.path.join(ann_path, name + '_' + mode + '.json')
    visual_allimages(ann_path, image_path, save_path, ratio)


def visual_allimages(anno_file, image_path, save_path, ratio=1):
    """visualize all images and annotations info"""
    rpc = COCO.COCO(anno_file)
    image_ids = rpc.getImgIds()
    images = []
    anns = {}
    for img_id in image_ids:
        idxs = rpc.getAnnIds(imgIds=[img_id])
        if idxs:
            images.append(img_id)
            anns[img_id] = idxs

    for img_id in images:
        file_name = rpc.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(image_path, file_name)
        annos = rpc.loadAnns(anns[img_id])
        img = cv2.imread(img_path)
        return visual_image(img, annos, save_path, ratio)


def visual_image(img, annos, save_path, ratio=None, height=None, width=None, name=None, score_threshold=0.01):
    """visualize image and annotations info"""
    h, w = img.shape[0], img.shape[1]
    if height is not None and width is not None and (height != h or width != w):
        img, annos = resize_image(img, annos, width, height)
    elif ratio not in (None, 1):
        img, annos = resize_image(img, annos, w * ratio, h * ratio)

    color_list = np.array(eval_cfg.color_list).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    colors = [(color_list[_]).astype(np.uint8) for _ in range(len(color_list))]
    colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 3)

    h, w = img.shape[0], img.shape[1]
    num_objects = len(annos)
    name_list = []
    id_list = []
    for class_name, class_id in rpc_class_name2id.items():
        name_list.append(class_name)
        id_list.append(class_id)

    for i in range(num_objects):
        ann = annos[i]
        bbox = rpc_box_to_bbox(ann['bbox'])
        cat_id = ann['category_id']
        if cat_id in id_list:
            get_id = id_list.index(cat_id)
            name = name_list[get_id]
            c = colors[get_id].tolist()
        if "score" in ann:
            score = ann["score"]
            if score < score_threshold:
                continue
            txt = '{}{:.2f}'.format(name, ann["score"])
            cat_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - cat_size[1] - 5)),
                          (int(bbox[0] + cat_size[0]), int(bbox[1] - 2)), c, -1)
            cv2.putText(img, txt, (int(bbox[0]), int(bbox[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        ct = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
        cv2.circle(img, ct, 2, c, thickness=-1, lineType=cv2.FILLED)
        bbox = np.array(bbox, dtype=np.int32).tolist()
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)

    if annos and "image_id" in annos[0]:
        img_id = annos[0]["image_id"]
    else:
        img_id = random.randint(0, 9999999)
    image_name = "cv_image_" + str(img_id) + ".png"
    cv2.imwrite("{}/{}".format(save_path, image_name), img)
