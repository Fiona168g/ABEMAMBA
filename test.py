#!/usr/bin/env python
"""
For evaluation
"""
import shutil
import SimpleITK as sitk
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from models.fewshot import FewShotSeg
from dataloaders.datasets import TestDataset
from dataloaders.dataset_specifics import *
from utils import *
from config import ex

import matplotlib
matplotlib.use('Agg')
import torch.nn.functional as F
import cv2
import numpy as np
import os
from PIL import Image
import time
from thop import profile



@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/interm_preds', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

        # Set up logger -> log to .txt
        file_handler = logging.FileHandler(os.path.join(f'{_run.observers[0].dir}', f'logger.log'))
        file_handler.setLevel('INFO')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        _log.handlers.append(file_handler)
        _log.info(f'Run "{_config["exp_str"]}" with ID "{_run.observers[0].dir[-1]}"')

    # Deterministic setting for reproduciablity.
    if _config['seed'] is not None:
        random.seed(_config['seed'])
        torch.manual_seed(_config['seed'])
        torch.cuda.manual_seed_all(_config['seed'])
        cudnn.deterministic = True

    # Enable cuDNN benchmark mode to select the fastest convolution algorithm.
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info(f'Create model...')
    model = FewShotSeg()
    model.cuda()
    model.load_state_dict(torch.load(_config['reload_model_path'], map_location='cpu'))
    
    
    ###########################FLOPS###########################################################
    def count_model_params_flops(model, support_image_s, support_fg_mask_s, query_image_s, query_mask):
        model.eval()
        with torch.no_grad():
            dummy_input = (support_image_s, support_fg_mask_s, query_image_s, query_mask)  # 确保传入 query_mask
            flops, params = profile(model, inputs=dummy_input)
    
        params_m = params / 1e6  # 转换为 M
        flops_g = flops / 1e9  # 转换为 GFLOPs
    
        print(f"Total Parameters: {params_m:.2f}M")
        print(f"Total FLOPs: {flops_g:.2f}GFLOPs")
        
        return params_m, flops_g


    _log.info(f'Load data...')
    data_config = {
        'data_dir': _config['path'][_config['dataset']]['data_dir'],
        'dataset': _config['dataset'],
        'n_shot': _config['n_shot'],
        'n_way': _config['n_way'],
        'n_query': _config['n_query'],
        'n_sv': _config['n_sv'],
        'max_iter': _config['max_iters_per_load'],
        'eval_fold': _config['eval_fold'],
        'min_size': _config['min_size'],
        'max_slices': _config['max_slices'],
        'supp_idx': _config['supp_idx'],
    }
    test_dataset = TestDataset(data_config)
    test_loader = DataLoader(test_dataset,
                             batch_size=_config['batch_size'],
                             shuffle=False,
                             num_workers=_config['num_workers'],
                             pin_memory=True,
                             drop_last=True)

    # Get unique labels (classes).
    labels = get_label_names(_config['dataset'])

    # Loop over classes.
    class_dice = {}
    class_iou = {}

    _log.info(f'Starting validation...')
    for label_val, label_name in labels.items():

        # Skip BG class.
        if label_name == 'BG':
            continue
        elif (not np.intersect1d([label_val], _config['test_label'])):
            continue

        _log.info(f'Test Class: {label_name}')

        # Get support sample + mask for current class.
        support_sample = test_dataset.getSupport(label=label_val, all_slices=False, N=_config['n_part'])

        test_dataset.label = label_val

        # Test.
        with torch.no_grad():
            model.eval()

            # Unpack support data.
            support_image = [support_sample['image'][[i]].float().cuda() for i in
                             range(support_sample['image'].shape[0])]  # n_shot x 3 x H x W, support_image is a list {3X(1, 3, 256, 256)}
            support_fg_mask = [support_sample['label'][[i]].float().cuda() for i in
                               range(support_sample['image'].shape[0])]  # n_shot x H x W

            # Loop through query volumes.
            scores = Scores()
            for i, sample in enumerate(test_loader):  # this "for" loops 4 times

                # Unpack query data.
                query_image = [sample['image'][i].float().cuda() for i in
                               range(sample['image'].shape[0])]  # [C x 3 x H x W] query_image is list {(C x 3 x H x W)}
                query_label = sample['label'].long()  # C x H x W
                query_id = sample['id'][0].split('image_')[1][:-len('.nii.gz')]

                # Compute output.
                # Match support slice and query sub-chunck.
                query_pred = torch.zeros(query_label.shape[-3:])
                C_q = sample['image'].shape[1]    # slice number of query img

                idx_ = np.linspace(0, C_q, _config['n_part'] + 1).astype('int')
                for sub_chunck in range(_config['n_part']):  # n_part = 3
                    support_image_s = [support_image[sub_chunck]]  # 1 x 3 x H x W
                    support_fg_mask_s = [support_fg_mask[sub_chunck]]  # 1 x H x W
                    query_image_s = query_image[0][idx_[sub_chunck]:idx_[sub_chunck + 1]]  # C' x 3 x H x W
                    query_pred_s = []
                    for i in range(query_image_s.shape[0]):
                    
                        start_time = time.time()
                        _pred_s, _, _, _, _, pred_ups = model([support_image_s], [support_fg_mask_s], [query_image_s[[i]]], _, train=False)  # 1 x 2 x H x W
                        query_pred_s.append(_pred_s)
                        
                        
                        count_model_params_flops(model, [support_image_s], [support_fg_mask_s], [query_image_s[[i]]], query_mask=None)
                        
                        
                        end_time = time.time()  # 结束计时
                        inference_time = end_time - start_time  # 计算单张图片推理时间
                        _log.info(f'Inference time for single image: {inference_time} seconds')
                        
                        
                        ###########################CAM#######################################
                        print("pred_ups", pred_ups.shape)
                        query_pred_np = pred_ups[0, 1].cpu().numpy()  # (H, W)
                        query_pred_np = (query_pred_np - query_pred_np.min()) / (query_pred_np.max() - query_pred_np.min())  # 归一化
                        
                        # ✅ 乘 255 后转换为 uint8，确保不是全 0
                        query_pred_np = (query_pred_np * 255).astype(np.uint8)
                        
                        # ✅ 生成彩色热力图
                        heatmap = cv2.applyColorMap(query_pred_np, cv2.COLORMAP_JET)  # (H, W, 3)
                        print("heatmap", heatmap.shape)  # (257, 257, 3)
                        
                        # 处理原始图像
                        original_image = query_image_s[[i]].cpu().numpy().squeeze()
                        original_image = np.transpose(original_image, (1, 2, 0))  # 变成 (H, W, 3)
                        original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min()) * 255
                        original_image = original_image.astype(np.uint8)
                        print("original_image", original_image.shape)
                        
                        # ✅ 叠加热力图
                        alpha = 0.5
                        superimposed_image = cv2.addWeighted(original_image, alpha, heatmap, 1 - alpha, 0)
                        
                        # ✅ 保存叠加后的图像
                        cam_filename = os.path.join(f'{_run.observers[0].dir}/interm_preds', f'CAM_{label_name}_{query_id}_subchunck{sub_chunck}_part{i}.png')
                        cv2.imwrite(cam_filename, superimposed_image)

                        
                    query_pred_s = torch.cat(query_pred_s, dim=0)
                    query_pred_s = query_pred_s.argmax(dim=1).cpu()  # C x H x W
                    query_pred[idx_[sub_chunck]:idx_[sub_chunck + 1]] = query_pred_s

                # Record scores.
                scores.record(query_pred, query_label)

                # Log.
                _log.info(
                    f'Tested query volume: {sample["id"][0][len(_config["path"][_config["dataset"]]["data_dir"]):]}.')
                _log.info(f'Dice score: {scores.patient_dice[-1].item()}')

                # Save predictions.
                file_name = os.path.join(f'{_run.observers[0].dir}/interm_preds',
                                         f'prediction_{query_id}_{label_name}.nii.gz')
                itk_pred = sitk.GetImageFromArray(query_pred)
                sitk.WriteImage(itk_pred, file_name, True)
                _log.info(f'{query_id} has been saved. ')

            # Log class-wise results
            class_dice[label_name] = torch.tensor(scores.patient_dice).mean().item()
            class_iou[label_name] = torch.tensor(scores.patient_iou).mean().item()
            _log.info(f'Test Class: {label_name}')
            _log.info(f'Mean class IoU: {class_iou[label_name]}')
            _log.info(f'Mean class Dice: {class_dice[label_name]}')

    _log.info(f'Final results...')
    _log.info(f'Mean IoU: {class_iou}')
    _log.info(f'Mean Dice: {class_dice}')

    def dict_Avg(Dict):
        L = len(Dict)  # 取字典中键值对的个数
        S = sum(Dict.values())  # 取字典中键对应值的总和
        A = S / L
        return A

    value = dict_Avg(class_dice)
    with open('results.txt', 'w') as file:
        file.write(str(value))

    _log.info(f'Whole mean Dice: {dict_Avg(class_dice)}')
    _log.info(f'End of validation.')
    return 1
