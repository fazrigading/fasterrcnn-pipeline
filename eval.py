"""
Run evaluation on a trained model to get mAP and class wise AP.

USAGE:
python eval.py --data data_configs/voc.yaml --split test --weights outputs/training/fasterrcnn_convnext_small_voc_15e_noaug/best_model.pth --model fasterrcnn_convnext_small
"""
from datasets import (
    create_valid_dataset, create_valid_loader
)
from models.create_fasterrcnn_model import create_model
from torch_utils import utils
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import MulticlassPrecisionRecallCurve
from pprint import pprint
from tqdm import tqdm

import torch
import argparse
import yaml
import torchvision
import time
import numpy as np
import matplotlib.pyplot as plt

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', 
        default='data_configs/test_image_config.yaml',
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '-s', '--split', 
        default='val',
        help='(optional) data split used for validation'
    )
    parser.add_argument(
        '-m', '--model', 
        default='fasterrcnn_resnet50_fpn',
        help='name of the model'
    )
    parser.add_argument(
        '-mw', '--weights', 
        default=None,
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    parser.add_argument(
        '-ims', '--imgsz', 
        default=640, 
        type=int, 
        help='image size to feed to the network'
    )
    parser.add_argument(
        '-w', '--workers', default=4, type=int,
        help='number of workers for data processing/transforms/augmentations'
    )
    parser.add_argument(
        '-b', '--batch', 
        default=8, 
        type=int, 
        help='batch size to load the data'
    )
    parser.add_argument(
        '-d', '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='show class-wise mAP'
    )
    parser.add_argument(
        '-st', '--square-training',
        dest='square_training',
        action='store_true',
        help='Resize images to square shape instead of aspect ratio resizing \
              for single image training. For mosaic training, this resizes \
              single images to square shape first then puts them on a \
              square canvas.'
    )
    args = vars(parser.parse_args())

    # Load the data configurations
    with open(args['data']) as file:
        data_configs = yaml.safe_load(file)

    # Validation settings and constants.
    if args['split'] == "test":
        VALID_DIR_IMAGES = data_configs['TEST_DIR_IMAGES']
        VALID_DIR_LABELS = data_configs['TEST_DIR_LABELS']
    else: # Else use the validation images.
        VALID_DIR_IMAGES = data_configs['VALID_DIR_IMAGES']
        VALID_DIR_LABELS = data_configs['VALID_DIR_LABELS']
    NUM_CLASSES = data_configs['NC']
    CLASSES = data_configs['CLASSES']
    NUM_WORKERS = args['workers']
    DEVICE = args['device']
    BATCH_SIZE = args['batch']

    # Model configurations
    IMAGE_SIZE = args['imgsz']

    # Load the pretrained model
    create_model = create_model[args['model']]
    if args['weights'] is None:
        try:
            model, coco_model = create_model(num_classes=NUM_CLASSES, coco_model=True)
        except:
            model = create_model(num_classes=NUM_CLASSES, coco_model=True)
        if coco_model:
            COCO_91_CLASSES = data_configs['COCO_91_CLASSES']
            valid_dataset = create_valid_dataset(
                VALID_DIR_IMAGES, 
                VALID_DIR_LABELS, 
                IMAGE_SIZE, 
                COCO_91_CLASSES, 
                square_training=args['square_training']
            )

    # Load weights.
    if args['weights'] is not None:
        model = create_model(num_classes=NUM_CLASSES, coco_model=False)
        checkpoint = torch.load(args['weights'], weights_only=True, map_location=DEVICE)    
        try:
            # Remove the 'module.' prefix from the keys if necessary
            state_dict = checkpoint['model_state_dict']
            if any(key.startswith('module.') for key in state_dict.keys()):
                print("Detected 'module.' prefix in state_dict keys. Removing prefix...")
                new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
                state_dict = new_state_dict

            # Attempt to load the state_dict
            model.load_state_dict(state_dict)
            print("Model loaded successfully.")
        except RuntimeError as e:
            # Check for specific error in loading state_dict
            if "Missing key(s) in state_dict" in str(e) or "Unexpected key(s) in state_dict" in str(e):
                print("Warning: State dict loading encountered an issue.")
                # Load state_dict with strict=False to ignore missing/unexpected keys
                model.load_state_dict(state_dict, strict=False)
                print("Model loaded with missing or unexpected keys ignored.")
            else:
                # If it's another error, raise it
                raise e
        del checkpoint

        valid_dataset = create_valid_dataset(
            VALID_DIR_IMAGES, 
            VALID_DIR_LABELS, 
            IMAGE_SIZE, 
            CLASSES,
            square_training=args['square_training']
        )
    model.to(DEVICE).eval()
    
    valid_loader = create_valid_loader(valid_dataset, BATCH_SIZE, NUM_WORKERS)

    @torch.inference_mode()
    def evaluate(
        model, 
        data_loader, 
        device, 
        out_dir=None,
        classes=None,
        colors=None
    ):
        metric = MeanAveragePrecision(class_metrics=args['verbose'])
        mcprc = MulticlassPrecisionRecallCurve(num_classes=NUM_CLASSES) # pr-curve feature
        n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Test:"

        target = []
        preds = []
        counter = 0
        for images, targets in tqdm(metric_logger.log_every(data_loader, 100, header), total=len(data_loader)):
            counter += 1
            images = list(img.to(device) for img in images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            with torch.no_grad():
                outputs = model(images)

            #####################################
            for i in range(len(images)):
                true_dict = dict()
                preds_dict = dict()
                true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
                true_dict['labels'] = targets[i]['labels'].detach().cpu()
                preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
                preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
                preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
                preds.append(preds_dict)
                target.append(true_dict)
            #####################################
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        torch.set_num_threads(n_threads)
        metric.update(preds, target)
        metric_summary = metric.compute()

        # PR-CURVE COMPUTE
        # Initialize an empty list for per-class scores
        pred_scores_list = []

        for pred in preds:
            # pred['labels'] is of shape (num_boxes,)
            # pred['scores'] is of shape (num_boxes,)
            num_boxes = pred['labels'].shape[0]
            # Create a one-hot tensor with zeros
            one_hot = torch.zeros((num_boxes, NUM_CLASSES), dtype=pred['scores'].dtype, device=pred['scores'].device)
            # Fill the predicted class column with the scores
            one_hot.scatter_(1, pred['labels'].unsqueeze(1), pred['scores'].unsqueeze(1))
            pred_scores_list.append(one_hot)

        pred_scores = torch.cat(pred_scores_list, dim=0)
        true_labels = torch.cat([target_entry['labels'] for target_entry in target], dim=0)
        mcprc.update(pred_scores, true_labels)
        # precision, recall, thresholds = mcprc.compute() # Disabled because no purpose?

        # PR-CURVE PLOT
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
        mcprc.plot(score=True, ax=ax)
        plot_data_split = "test" if args['split'] == "test" else "val"
        fig.savefig(f"pr_curve_{plot_data_split}.png", dpi=250)
        plt.close(fig)

        return metric_summary

    stats = evaluate(
        model, 
        valid_loader, 
        device=DEVICE,
        classes=CLASSES,
    )

    print('\n')
    pprint(stats)
    if args['verbose']:
        print('\n')
        pprint(f"Classes: {CLASSES}")
        print('\n')
        print('AP / AR per class')
        empty_string = ''
        if len(CLASSES) > 2: 
            num_hyphens = 73
            print('-'*num_hyphens)
            print(f"|    | Class{empty_string:<16}| AP{empty_string:<18}| AR{empty_string:<18}|")
            print('-'*num_hyphens)
            class_counter = 0
            for i in range(0, len(CLASSES)-1, 1):
                class_counter += 1
                print(f"|{class_counter:<3} | {CLASSES[i+1]:<20} | {np.array(stats['map_per_class'][i]):.3f}{empty_string:<15}| {np.array(stats['mar_100_per_class'][i]):.3f}{empty_string:<15}|")
            print('-'*num_hyphens)
            print(f"|Avg{empty_string:<23} | {np.array(stats['map']):.3f}{empty_string:<15}| {np.array(stats['mar_100']):.3f}{empty_string:<15}|")
        else:
            num_hyphens = 62
            print('-'*num_hyphens)
            print(f"|Class{empty_string:<10} | AP{empty_string:<18}| AR{empty_string:<18}|")
            print('-'*num_hyphens)
            print(f"|{CLASSES[1]:<15} | {np.array(stats['map']):.3f}{empty_string:<15}| {np.array(stats['mar_100']):.3f}{empty_string:<15}|")
            print('-'*num_hyphens)
            print(f"|Avg{empty_string:<12} | {np.array(stats['map']):.3f}{empty_string:<15}| {np.array(stats['mar_100']):.3f}{empty_string:<15}|")