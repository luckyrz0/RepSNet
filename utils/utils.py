from cv2 import imwrite
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import colorsys
import random

cmap = plt.get_cmap("jet")


def binarize(x):
    '''
    convert multichannel (multiclass) instance segmetation tensor
    to binary instance segmentation (bg and nuclei),
    :param x: B*B*C (for PanNuke 256*256*5 )
    :return: Instance segmentation
    '''
    out = np.zeros([x.shape[0], x.shape[1]])
    count = 1
    for i in range(x.shape[2]):
        x_ch = x[:, :, i]
        unique_vals = np.unique(x_ch)
        unique_vals = unique_vals.tolist()
        unique_vals.remove(0)
        for j in unique_vals:
            x_tmp = x_ch == j
            x_tmp_c = 1 - x_tmp
            out *= x_tmp_c
            out += count * x_tmp
            count += 1
    out = out.astype('int32')
    return out


def colorize(ch, vmin, vmax):
    """
    Will clamp value value outside the provided range to vmax and vmin
    """
    ch = np.squeeze(ch.astype("float32"))
    ch[ch > vmax] = vmax  # clamp value
    ch[ch < vmin] = vmin
    ch = (ch - vmin) / (vmax - vmin + 1.0e-16)
    # take RGB from RGBA heat map
    ch_cmap = (cmap(ch)[..., :3] * 255).astype("uint8")
    # ch_cmap = center_pad_to_shape(ch_cmap, aligned_shape)
    return ch_cmap


def random_colors(N, bright=True):
    """Generate random colors.
        
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def visualize_instances_map(input_image, inst_map, type_map=None, type_colour=None, line_thickness=1, with_over=True):

    if with_over:
        overlay = np.copy((input_image).astype(np.uint8))
    else:
        overlay = np.zeros_like(input_image).astype(np.uint8)

    inst_list = list(np.unique(inst_map))  # get list of instances
    inst_list.remove(0)  # remove background

    inst_rng_colors = random_colors(len(inst_list))
    inst_rng_colors = np.array(inst_rng_colors) * 255
    inst_rng_colors = inst_rng_colors.astype(np.uint8)

    for inst_idx, inst_id in enumerate(inst_list):
        inst_map_mask = np.array(inst_map == inst_id, np.uint8)  # get single object

        y1, y2, x1, x2 = get_bounding_box(inst_map_mask)
        y1 = y1 - 2 if y1 - 2 >= 0 else y1
        x1 = x1 - 2 if x1 - 2 >= 0 else x1
        x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
        y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2
        inst_map_crop = inst_map_mask[y1:y2, x1:x2]
        contours_crop = cv2.findContours(inst_map_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # only has 1 instance per map, no need to check #contour detected by opencv
        contours_crop = np.squeeze(contours_crop[0][0].astype("int32"))  # * opencv protocol format may break
        contours_crop = contours_crop.reshape(-1, 2)  # by xsc

        contours_crop += np.asarray([[x1, y1]])  # index correction
        if type_map is not None:
            type_map_crop = type_map[y1:y2, x1:x2]
            type_id = np.unique(type_map_crop).max()  # non-zero
            inst_colour = type_colour[type_id]
        else:
            inst_colour = (inst_rng_colors[inst_idx]).tolist()
        cv2.drawContours(overlay, [contours_crop], -1, inst_colour, line_thickness)
    return overlay


def viz_map(out, feed_dict):

    img = feed_dict["img"].data.numpy()

    H, W = img.shape[1:3]

    pred_viz_list = []
    ture_viz_list = []

    if "type_map" in out:
        type_pred = out["type_map"].to("cpu").data.numpy()
        type_pred = np.argmax(type_pred, axis=1)
        type_true = feed_dict["type_map"].data.numpy()

        num_types = out["type_map"].shape[1]

        pred_viz_list.append(colorize(type_pred, 0, num_types).reshape(-1, H, W, 3))
        ture_viz_list.append(colorize(type_true, 0, num_types).reshape(-1, H, W, 3))

    if "nucleus_map" in out:
        nucleus_pred = out["nucleus_map"].to("cpu").data.numpy()
        nucleus_pred = np.argmax(nucleus_pred, axis=1)

        nucleus_true = (type_true > 0).astype("int32")

        pred_viz_list.append(colorize(nucleus_pred, 0, 1).reshape(-1, H, W, 3))
        ture_viz_list.append(colorize(nucleus_true, 0, 1).reshape(-1, H, W, 3))

    if "boundary_probability_map" in out:
        boundary_pred = out["boundary_probability_map"].to("cpu").data.numpy()
        boundary_pred = np.argmax(boundary_pred, axis=1)
        boundary_true = feed_dict["boundary_map"].data.numpy()

        pred_viz_list.append(colorize(boundary_pred > 0, 0, 1).reshape(-1, H, W, 3))
        ture_viz_list.append(colorize(boundary_true == 1, 0, 1).reshape(-1, H, W, 3))

    if "boundary_map" in out:
        boundary_pred = out["boundary_map"].to("cpu").data.numpy()
        boundary_true = feed_dict["boundary_map"].data.numpy()

        pred_viz_list.append(colorize(boundary_pred >= 4, 0, 1).reshape(-1, H, W, 3))
        ture_viz_list.append(colorize(boundary_true == 1, 0, 1).reshape(-1, H, W, 3))

    if "ann_map" in out:
        ann_pred = out["ann_map"].astype("int32")
        ann_true = feed_dict["ann"].data.numpy()

        color_list = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 0, 255], [255, 255, 255], [0, 0, 0]]

        instance_img_pred = [visualize_instances_map(img[i], ann_pred[i, :, :, 0], ann_pred[i, :, :, 1], color_list) for i in range(img.shape[0])]
        instance_img_pred = np.array(instance_img_pred)
        pred_viz_list.append(instance_img_pred)

        instance_img_pred = [visualize_instances_map(img[i], ann_pred[i, :, :, 0], with_over=False, line_thickness=-1) for i in range(img.shape[0])]
        instance_img_pred = np.array(instance_img_pred)
        pred_viz_list.append(instance_img_pred)

        instance_img_true = [visualize_instances_map(img[i], ann_true[i, :, :, 0], type_true[i], color_list) for i in range(img.shape[0])]
        instance_img_true = np.array(instance_img_true)
        ture_viz_list.append(instance_img_true)

        instance_img_true = [visualize_instances_map(img[i], ann_true[i, :, :, 0], with_over=False, line_thickness=-1) for i in range(img.shape[0])]
        instance_img_true = np.array(instance_img_true)
        ture_viz_list.append(instance_img_true)

    if "boundary_distance_map" in out:
        boundary_distance_pred = out["boundary_distance_map"].to("cpu").data.numpy()
        boundary_distance_true = feed_dict["boundary_distance_map"].data.numpy()

        for i in range(boundary_distance_true.shape[-1]):
            pred_viz_list.append(colorize(boundary_distance_pred[:, i], 0, boundary_distance_pred[:, i].max()).reshape(-1, H, W, 3))
            ture_viz_list.append(colorize(boundary_distance_true[..., i], 0, boundary_distance_true[..., i].max()).reshape(-1, H, W, 3))

    if "hv_map" in out:
        hv_pred = out["hv_map"].to("cpu").data.numpy()
        hv_true = feed_dict["hv_map"].data.numpy()

        for i in range(hv_true.shape[-1]):
            pred_viz_list.append(colorize(hv_pred[:, i], -1, 1).reshape(-1, H, W, 3))
            ture_viz_list.append(colorize(hv_true[..., i], -1, 1).reshape(-1, H, W, 3))

    if "marker_map" in out:
        marker_pred = out["marker_map"]
        pred_viz_list.append(colorize(marker_pred, 0, 1).reshape(-1, H, W, 3))
        ture_viz_list.append(img)

    output_img = np.concatenate([np.concatenate(ture_viz_list, axis=1), np.concatenate(pred_viz_list, axis=1)], axis=2)
    return output_img
