import os
import numpy as np
from lib.model_utils.config import config, dataset_config, eval_config, net_config
from lib.dataset import RPC


def preprocess(dataset_path, preprocess_path):
    """preprocess input images"""
    meta_path = os.path.join(preprocess_path, "meta/meta")
    result_path = os.path.join(preprocess_path, "data")
    if not os.path.exists(meta_path):
        os.makedirs(os.path.join(preprocess_path, "meta/meta"))
    if not os.path.exists(result_path):
        os.makedirs(os.path.join(preprocess_path, "data"))
    rpc = RPC(dataset_config, run_mode="val", net_opt=net_config)
    rpc.init(dataset_path, keep_res=False)
    dataset = rpc.create_eval_dataset()
    name_list = []
    meta_list = []
    i = 0
    for data in dataset.create_dict_iterator(num_epochs=1):
        img_id = data['image_id'].asnumpy().reshape((-1))[0]
        image = data['image'].asnumpy()
        for scale in eval_config.multi_scales:
            image_preprocess, meta = rpc.pre_process_for_test(image, img_id, scale)
        evl_file_name = "eval2017_image" + "_" + str(img_id) + ".bin"
        evl_file_path = result_path + "/" + evl_file_name
        image_preprocess.tofile(evl_file_path)
        meta_file_path = os.path.join(preprocess_path + "/meta/meta", str(img_id) + ".txt")
        with open(meta_file_path, 'w+') as f:
            f.write(str(meta))
        name_list.append(img_id)
        meta_list.append(meta)
        i += 1
        print(f"preprocess: no.[{i}], img_name:{img_id}")
    np.save(os.path.join(preprocess_path + "/meta", "name_list.npy"), np.array(name_list))
    np.save(os.path.join(preprocess_path + "/meta", "meta_list.npy"), np.array(meta_list))


if __name__ == '__main__':
    preprocess(config.val_data_dir, config.predict_dir)
