from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


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
    i = 0
    N = len(dataloader)
    label_map = {0: 'background', 1: 'car', 2: 'truck', 3: 'bus', 4: 'motorcycle', 5: 'bicycle', 6: 'scooter', 7: 'person', 8: 'rider'}
    classes = list(label_map.values())[1:]
    print(classes)
    num_boxes_per_class = np.empty((N, len(label_map)-1))
    
    #dict_keys(['image', 'boxes', 'labels', 'width', 'height', 'image_id'])
    
    for batch in tqdm(dataloader):
        # Remove the two lines below and start analyzing :D
        #print("The keys in the batch are:", batch.keys())
        #print(batch['labels'])
        #print(batch['labels'].shape[1])
        #print(batch['labels'])
        num_boxes_per_class[i, 0] = batch['labels'].shape[1]
        for j in range(1, 8+1):
            #print("Type:", type(np.array(batch['labels'])))
            num_boxes_per_class[i, j-1] = np.sum(np.array(batch['labels'])==j)
        
        i = i+1
        #if i == 6:
        #    break
    
    plt.figure()
    temp = np.kron(num_boxes_per_class, np.ones(200))
    plt.imshow(temp)
    plt.xticks(range(100, 1600, 200), ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'])
    
    plt.figure()
    plt.bar(range(len(classes)), np.sum(num_boxes_per_class, axis=0))
    plt.xticks(range(len(classes)), ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'])
    
    plt.waitforbuttonpress()
    exit()


def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    analyze_something(dataloader, cfg)


if __name__ == '__main__':
    main()
