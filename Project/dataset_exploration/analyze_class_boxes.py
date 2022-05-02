from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tops.torch_utils import set_seed
import torch


def get_config(config_path):
    cfg = LazyConfig.load(config_path)
    cfg.train.batch_size = 1
    return cfg


def get_dataloader(cfg, dataset_to_visualize):
    if dataset_to_visualize == "train":
        # Remove GroundTruthBoxesToAnchors transform
        cfg.data_train.dataset.transform.transforms = cfg.data_train.dataset.transform.transforms[:-1]
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader


def analyze_something(dataloader, cfg):
    N = len(dataloader)
    label_map = {0: 'background', 1: 'car', 2: 'truck', 3: 'bus', 4: 'motorcycle', 5: 'bicycle', 6: 'scooter', 7: 'person', 8: 'rider'}
    
    # Change class below in order to analyze a different class
    class_label = 'bus'
    cropped_imgs = []
    
    w = 1024
    h = 128
    img = np.array([])
    for i, batch in enumerate(tqdm(dataloader)):
        for j, label_num in enumerate(batch['labels'][0]):
            label = label_map[label_num.item()]
            if label == 'bus':
                box = batch['boxes'][0][j].clone().detach()
                box[[0, 2]] *= w
                box[[1, 3]] *= h
                box = box.int()
                img = batch['image'][0][:,box[1]:box[3],box[0]:box[2]].clone().detach()
                cropped_imgs.append(img)

    # Display first n images of that class
    for i in range(15):
        image_pixel_values = (cropped_imgs[i] * 255).byte()
        image_h_w_c_format = image_pixel_values.permute(1, 2, 0)
        image = image_h_w_c_format.cpu().numpy()
        (h, w) = image.shape[:2]
        scaler = 400/h
        image = cv2.resize(image, (int(scaler*w), int(scaler*h)), interpolation = cv2.INTER_AREA)
        cv2.imshow(f'{class_label}{i}', image)
    
    cv2.waitKey()
    
        
    
    plt.figure()
    
    # plt.waitforbuttonpress()
    plt.savefig("dataset_exploration/class_boxes.png")


def main():
    set_seed(42)
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    analyze_something(dataloader, cfg)


if __name__ == '__main__':
    main()
