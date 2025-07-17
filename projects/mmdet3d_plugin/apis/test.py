import os
import shutil
import tempfile
import time
import cv2
import torchvision
import time

from PIL import Image

import mmcv
import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt

from mmcv.runner import get_dist_info
from mmdet.utils import get_root_logger
from projects.mmdet3d_plugin.metrics import IntersectionOverUnion

import json


def show_seg(labels, car_img):

    PALETTE = [[255, 255, 255], [220, 20, 60], [0, 0, 128], [0, 100, 0],
               [128, 0, 0], [64, 0, 128], [64, 0, 192], [192, 128, 64],
               [192, 192, 128], [64, 64, 128], [128, 0, 192], [192, 0, 64]]
    mask_colors = np.array(PALETTE)
    img = np.zeros((200, 400, 3))

    for index, mask_ in enumerate(labels):
        color_mask = mask_colors[index]
        mask_ = mask_.astype(bool)
        img[mask_] = color_mask

    # 这里需要水平翻转，因为这样才可以保证与在图像坐标系下，与习惯相同

    img = np.flip(img, axis=0)
    # 可视化小车
    car_img = np.where(car_img == [0, 0, 0], [255, 255, 255], car_img)[16: 84, 5:, :]
    car_img = cv2.resize(car_img.astype(np.uint8), (30, 16))
    img[img.shape[0] // 2 - 8: img.shape[0] // 2 + 8, img.shape[1] // 2 - 15: img.shape[1] // 2 + 15, :] = car_img

    return img

def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    cv2_style=False):

    logger = get_root_logger()

    # multi-task settings
    test_mode = data_loader.dataset.test_submission

    map_enable = True
    if test_mode:
        map_enable = False

    if map_enable:
        num_map_class = 6
        semantic_map_iou_val = IntersectionOverUnion(num_map_class)
        semantic_map_iou_val = semantic_map_iou_val.cuda()

    model.eval()
    results = []
    inference_times = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        start = time.time()

        with torch.no_grad():
            in_data = {i: j for i, j in data.items() if 'img' in i}
            result = model(return_loss=False, rescale=True, **in_data)

        batch_size = len(result)
        assert batch_size == 1, 'val step batch size must set 1!'
        show_mask_gt = True

        torch.cuda.synchronize()
        end = time.time()
        elapsed = end - start
        inference_times.append(elapsed)

        if (result[0]['seg_preds'] is not None) and (show or out_dir):
            images = dict()
            N = min(result[0]['seg_preds'].shape[0], 4)
            images['pred'] = result[0]['seg_preds'][:N]

            processed_images = []
            spacing = 8
            for k in images:
                if isinstance(images[k], np.ndarray):
                    images[k] = torch.from_numpy(images[k])
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                grid = (grid > 0).cpu().numpy().astype(np.uint8)
                processed_images.append(grid)

            if len(processed_images) > 0:

                # add blank gap between images
                images_with_spacing = []
                for img in processed_images:
                    # img = np.transpose(img, (2, 0, 1))  # [512, 512, 4] -> [4, 512, 512]
                    horizontal_mask = np.concatenate(img, axis=1)   # [4, 512, 512] -> [512, 2048]
                    rgb_mask = np.stack([horizontal_mask] * 3, axis=2)  # [512, 2048] -> [512, 2048, 3] to show in RGB
                    img = Image.fromarray(rgb_mask * 255)  # convert to PIL Image, rescale to [0, 255]
                    images_with_spacing.append(img)
                    _, img_width = horizontal_mask.shape
                    blank_image = np.ones((spacing, img_width, 3), dtype=np.uint8) * 255  # blank gap
                    images_with_spacing.append(blank_image)
                # remove last blank gap of the last image
                images_with_spacing = images_with_spacing[:-1]

                stacked_image = np.vstack(images_with_spacing)  # stack vertically

                # save the combined image
                filename = "result-{:05}.png".format(i)
                path = os.path.join(out_dir, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(stacked_image).save(path)

        if result[0]['pts_bbox'] == None:
            pass

        else:
            results.extend([dict(pts_bbox=result[0]['pts_bbox'])])

        if result[0]['seg_preds'] == None:
            map_enable = False

        if map_enable:
            pred_semantic_indices = result[0]['seg_preds']
            # pred = onehot_encoding(pred)
            # num_cls = pred.shape[1]
            # indices = torch.arange(0, num_cls).reshape(-1, 1, 1).to(pred.device)
            # pred_semantic_indices = torch.sum(pred * indices, axis=1).int()
            target_semantic_indices = data['semantic_indices'][0].cuda()

            semantic_map_iou_val(pred_semantic_indices,
                                 target_semantic_indices)

        for _ in range(batch_size):
            prog_bar.update()

    # Calculate average inference time and FPS
    total_time = sum(inference_times)
    num_samples = len(inference_times)
    avg_time = total_time / num_samples
    fps = 1.0 / avg_time
    print(f"\n✅ Average time per sample = {avg_time:.4f} s, FPS = {fps:.2f}")

    if map_enable:
        import prettytable as pt
        scores = semantic_map_iou_val.compute()
        mIoU = scores.mean()
        tb = pt.PrettyTable()
        categories = ['drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 'lane_divider']
        tb.field_names = ['Validation num'] + categories + ['mIoU']
        class_scores = [round(s.item(), 6) for s in scores]
        tb.add_row([len(dataset)] + class_scores + [round(mIoU.item(), 4)])

        print('\n')
        print(tb)
        logger.info(tb)

        seg_dict = {'Validation_num': len(dataset)}
        seg_dict.update({
            name: round(score.item(), 4)
            for name, score in zip(categories, scores)
        })
        seg_dict['mIoU'] = round(mIoU.item(), 4)

        with open('segmentation_result.json', 'a') as f:
            f.write(json.dumps(seg_dict) + '\n')

    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """

    logger = get_root_logger()
    # multi-task settings
    test_mode = data_loader.dataset.test_submission
    map_enable = True
    if test_mode:
        map_enable = False

    if map_enable:
        num_map_class = 4
        semantic_map_iou_val = IntersectionOverUnion(num_map_class)
        semantic_map_iou_val = semantic_map_iou_val.cuda()

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()

    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            in_data = {i: j for i, j in data.items() if 'img' in i}
            result = model(return_loss=False, rescale=True, **in_data)

            batch_size = len(result)
            # print(result)
            if result[0]['pts_bbox'] == None:
                pass

            else:
                results.extend([dict(pts_bbox=result[0]['pts_bbox'])])

        if result[0]['seg_preds'] == None:
            map_enable = False

        if map_enable:
            pred = result[0]['seg_preds']
            pred = onehot_encoding(pred)
            num_cls = pred.shape[1]
            indices = torch.arange(0, num_cls).reshape(-1, 1, 1).to(pred.device)
            pred_semantic_indices = torch.sum(pred * indices, axis=1).int()
            target_semantic_indices = data['semantic_indices'][0].cuda()

            semantic_map_iou_val(pred_semantic_indices,
                                 target_semantic_indices)

        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()

    if map_enable:
        import prettytable as pt
        scores = semantic_map_iou_val.compute()
        mIoU = sum(scores[1:]) / (len(scores) - 1)
        if rank == 0:
            tb = pt.PrettyTable()
            tb.field_names = ['Validation num', 'Divider', 'Pred Crossing', 'Boundary', 'mIoU']
            tb.add_row([len(dataset), round(scores[1:].cpu().numpy()[0], 4),
                        round(scores[1:].cpu().numpy()[1], 4), round(scores[1:].cpu().numpy()[2], 4),
                        round(mIoU.cpu().numpy().item(), 4)])
            print('\n')
            #print(tb)
            logger.info(tb)

            seg_dict = dict(
                Validation_num=len(dataset),
                Divider=round(scores[1:].cpu().numpy()[0], 4),
                Pred_Crossing=round(scores[1:].cpu().numpy()[1], 4),
                Boundary=round(scores[1:].cpu().numpy()[2], 4),
                mIoU=round(mIoU.cpu().numpy().item(), 4)
            )

            with open('segmentation_result.json', 'a') as f:
                f.write(json.dumps(str(seg_dict)) + '\n')

    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, os.path.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = os.path.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        # for res in zip(*part_list):
        for res in part_list:
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    return collect_results_cpu(result_part, size)

