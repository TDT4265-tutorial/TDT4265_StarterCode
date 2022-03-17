import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    
    # Area of overlap = Area of the "inner" rectangle given by the boxes
    # area_of_overlap = (np.minimum(prediction_box[2], gt_box[2]) - np.maximum(prediction_box[0], gt_box[0])) \
        # * (np.minimum(prediction_box[3], gt_box[3]) - np.maximum(prediction_box[1], gt_box[1]))
    area_of_overlap = np.maximum(np.minimum(prediction_box[2], gt_box[2]) - np.maximum(prediction_box[0], gt_box[0]), 0) \
        * np.maximum(np.minimum(prediction_box[3], gt_box[3]) - np.maximum(prediction_box[1], gt_box[1]),0)
    
    # Area of union = The area of both boxes minus the overlap
    area_of_union = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1]) \
        + (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]) \
        - area_of_overlap

    # IoU = Area of overlap / Area of union
    # iou = np.maximum(area_of_overlap / area_of_union, 0)
    iou = area_of_overlap / area_of_union

    assert iou >= 0 and iou <= 1, "iou not between 0 and 1" 
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
   
    if num_tp + num_fp == 0:
        return 1
    else:
        return num_tp / (num_tp + num_fp)


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """

    if num_tp + num_fn == 0:
        return 0
    else:
        return num_tp / (num_tp + num_fn)


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold

    # prediction_boxes_matched = np.array([])
    # gt_boxes_matched = np.array([])
    prediction_boxes_matched = []
    gt_boxes_matched = []
    ious = np.array([])

    for gt_box in gt_boxes:

        # Denne if-setningen "fikser" en bug på en ikke ideell måte
        if(len(prediction_boxes) > 0):

            for pred_box in prediction_boxes:
                # Finding IoU for all prediction boxes with the specified gt_box
                ious = np.append(ious, calculate_iou(pred_box, gt_box))

            # print('For gt_box:' , gt_box, 'The best IoU is:', np.amax(ious))
            
            # Checking if the maximum value of the IoU list is bigger than the treshold
            if(np.amax(ious) >= iou_threshold):
                # gt_boxes_matched = np.append(gt_boxes_matched, gt_box)
                gt_boxes_matched.append(gt_box)
                # prediction_boxes_matched = np.append(prediction_boxes_matched, prediction_boxes[np.argmax(ious)])
                prediction_boxes_matched.append(prediction_boxes[np.argmax(ious)])
                prediction_boxes = np.delete(prediction_boxes, np.argmax(ious), axis=0)

        ious = np.array([])

    # print('Prediction_boxes:', prediction_boxes)
    # print('Prediction_boxes_matched:', prediction_boxes_matched)
    # print('Gt_boxes_matched:', gt_boxes_matched)
    # return prediction_boxes_matched, gt_boxes_matched
    return np.array(prediction_boxes_matched), np.array(gt_boxes_matched)


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """

    # True positives = prediction boxes that are over the treshold len(prediction_boxes_matched)
    # False positives = prediction boxes that are not over the threshold
    # False negatives = 

    result = {
        'true_pos': 0,
        'false_pos': 0,
        'false_neg': 0
    }

    res1, res2 = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)

    # print("test:", res1)

    result['true_pos'] = len(res1)
    result['false_pos'] = len(prediction_boxes) - len(res1)
    result['false_neg'] = len(gt_boxes) - len(res2)

    return result


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """

    TP = 0
    FP = 0
    FN = 0

    for i in range(len(all_prediction_boxes)):
        # print('all_prediction_boxes:', all_prediction_boxes[i])
        # print('all_gt_boxes:', all_gt_boxes[i])
        result = calculate_individual_image_result(all_prediction_boxes[i], all_gt_boxes[i], iou_threshold)

        TP += result['true_pos']
        FP += result['false_pos']
        FN += result['false_neg']

    return (calculate_precision(TP, FP, FN), calculate_recall(TP, FP, FN))


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE

    precisions = [] 
    recalls = []
    confidence_prediction_boxes = []

    for confidence_threshold in confidence_thresholds:

        for i in range(len(confidence_scores)):
            confidence_prediction_boxes.append(all_prediction_boxes[i][confidence_scores[i] >= confidence_threshold,:])

        precision_recall = calculate_precision_recall_all_images(confidence_prediction_boxes, all_gt_boxes, iou_threshold)

        precision = precision_recall[0]
        recall = precision_recall[1]

        precisions.append(precision)
        recalls.append(recall)

        confidence_prediction_boxes = []


    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    average_precision = []
    precision_recall_over_thershold = []

    for recall_level in recall_levels:

        for precision, recall in zip(precisions, recalls):

            if recall >= recall_level:
                precision_recall_over_thershold.append(precision)
        
        if len(precision_recall_over_thershold) > 0:
            average_precision.append(max(precision_recall_over_thershold))
        else:
            average_precision.append(0)

        precision_recall_over_thershold = []
    
    average_precision = np.mean(average_precision)
    # print('average_precision:', average_precision)
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
