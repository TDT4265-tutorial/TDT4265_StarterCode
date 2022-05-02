from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tops.torch_utils import set_seed


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


def analyze_boxes(dataloader, cfg):
    N = len(dataloader)
    label_map = {0: 'background', 1: 'car', 2: 'truck', 3: 'bus', 4: 'motorcycle', 5: 'bicycle', 6: 'scooter', 7: 'person', 8: 'rider'}
    classes = list(label_map.values())[1:]
    
    box_size_per_class = np.empty(len(classes))
    num_boxes_per_class = np.zeros((N, len(label_map)-1))

    #dict_keys(['image', 'boxes', 'labels', 'width', 'height', 'image_id'])
    for i, batch in enumerate(tqdm(dataloader)):
        for k, box in enumerate(batch['boxes'][0]):
            box_size = (box[2] - box[0])*batch['width'] * (box[3] - box[1])*batch['height']
            box_size_per_class[batch['labels'][0][k]-1] += box_size

        for j in range(1, 8+1):
            num_boxes_per_class[i, j-1] = np.sum(np.array(batch['labels'])==j)

    boxes_per_class = np.sum(num_boxes_per_class, axis=0)
    box_size_per_class /= np.where(boxes_per_class==0, 1, boxes_per_class)

    plt.figure()
    bars = plt.bar(range(len(classes)), box_size_per_class)
    bar_height_max = 0
    for bar in bars:
        height = bar.get_height()
        if height > bar_height_max: bar_height_max = height
        plt.text(bar.get_x() + bar.get_width()/2., height+10,
                '%d' % int(height),
                ha='center', va='bottom')
    
    plt.xticks(range(len(classes)), classes)
    plt.ylabel("Avereage box size")
    plt.ylim(0, bar_height_max*1.1)
    
    # plt.waitforbuttonpress()
    plt.savefig("dataset_exploration/boxes.png")


def main():
    set_seed(42)
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    analyze_boxes(dataloader, cfg)


if __name__ == '__main__':
    main()
