import cv2
import numpy as np
import torch
from ultralytics.engine.results import Results
def get_color_by_obj_id(obj_id):
    """
    Get consistent color for a given object ID.
    
    Args:
        obj_id (int): Object ID
        
    Returns:
        tuple: BGR color tuple
    """
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (0, 128, 255),  # Orange-ish
        (128, 255, 0),  # Lime
        (255, 128, 0),  # Dark Orange
        (0, 255, 128),  # Spring Green
    ]
    return colors[obj_id % len(colors)]

def read_img(image_path, image_size=None):
    """
    Read an image from the specified path.
    
    Args:
        image_path (str): Path to the image file.
        image_size (int, optional): Target size to resize the image. If None, use original size.
        
    Returns:
        np.ndarray: Image array.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    if image_size is not None:
        img = cv2.resize(img, (image_size, image_size))
    return img





def get_imgs_labels_from_labelme_json(json_path, image_path, prompt_style="mask_prompt", image_size=None):
    """
    Extract prompts and labels from a Labelme JSON file with different prompt styles.
    
    Args:
        json_path (str): Path to the Labelme JSON file.
        image_path (str): Path to the image file.
        prompt_style (str): Type of prompt to generate. Options: 'mask_prompt', 'bbox_prompt', 'point_prompt'
        image_size (int, optional): Target size to resize the image and prompts. If None, use original size.
        
    Returns:
        tuple: (img, (prompts, labels)) where:
            - img (np.ndarray): The processed image
            - prompts (list): List of prompts in the specified style:
                - For 'mask_prompt': List of binary masks (2D numpy arrays)
                - For 'bbox_prompt': List of bounding boxes [x1, y1, x2, y2]
                - For 'point_prompt': List of center points [x, y]
            - labels (list): List of integer labels corresponding to each prompt
    """
    import json
    
    # Validate prompt_style
    valid_styles = ['mask_prompt', 'bbox_prompt', 'point_prompt']
    if prompt_style not in valid_styles:
        raise ValueError(f"prompt_style must be one of {valid_styles}, got '{prompt_style}'")
    
    # Load JSON data with encoding detection
    import chardet
    
    # First, detect the encoding
    with open(json_path, 'rb') as f:
        raw_data = f.read()
        encoding_result = chardet.detect(raw_data)
        detected_encoding = encoding_result['encoding'] or 'utf-8'
    
    # Then load with detected encoding
    try:
        with open(json_path, 'r', encoding=detected_encoding) as f:
            data = json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError):
        # If detected encoding fails, try common encodings
        for encoding in ['utf-8', 'gbk', 'gb2312', 'ascii', 'latin-1']:
            try:
                with open(json_path, 'r', encoding=encoding) as f:
                    data = json.load(f)
                print(f"Successfully loaded JSON with encoding: {encoding}")
                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        else:
            raise ValueError(f"Cannot decode JSON file {json_path} with any common encoding")
    
    # Read image - get original size first
    img = read_img(image_path, image_size=None)  # Don't resize yet
    original_height, original_width = img.shape[:2]
    
    prompts = []
    labels = []
    
    # Process each shape in the JSON
    for shape in data['shapes']:

        # Get polygon points
        points = np.array(shape['points'], dtype=np.float32)
        
        if prompt_style == "mask_prompt":
            if shape["shape_type"] in ["polygon"]:
                # Create binary mask for this polygon
                mask = np.zeros((original_height, original_width), dtype=np.uint8)
                points_int = points.astype(np.int32)
                cv2.fillPoly(mask, [points_int], 255)
                # Convert to boolean mask (0 or 1)
                mask = (mask > 0).astype(np.uint8)
                prompts.append(mask)    
                labels.append(int(shape['label'])) 
                
        elif prompt_style == "bbox_prompt":
            if shape["shape_type"]  in ["polygon","rectangle"]:
                # Calculate bounding box from polygon
                min_x = int(np.min(points[:, 0]))
                max_x = int(np.max(points[:, 0]))
                min_y = int(np.min(points[:, 1]))
                max_y = int(np.max(points[:, 1]))
                bbox = [min_x, min_y, max_x, max_y]
                prompts.append(bbox)
                labels.append(int(shape['label']))
        elif prompt_style == "point_prompt":
            if shape["shape_type"] == 'point':
                # Calculate center point from polygon
                center_x = int(np.mean(points[:, 0]))
                center_y = int(np.mean(points[:, 1]))
                center_point = [center_x, center_y]
                prompts.append(center_point)
                labels.append(int(shape['label']))
    
    # Resize image and prompts if image_size is specified
    if image_size is not None:
        scale_x = image_size / original_width
        scale_y = image_size / original_height
        img = cv2.resize(img, (image_size, image_size))
        
        if prompt_style == "mask_prompt":
            # Resize each mask
            resized_prompts = []
            for mask in prompts:
                resized_mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
                # Ensure binary values after resize
                resized_mask = (resized_mask > 0.5).astype(np.uint8)
                resized_prompts.append(resized_mask)
            prompts = resized_prompts
            
        elif prompt_style == "bbox_prompt":
            # Scale bounding boxes
            scaled_prompts = []
            for bbox in prompts:
                x1, y1, x2, y2 = bbox
                scaled_bbox = [
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y)
                ]
                scaled_prompts.append(scaled_bbox)
            prompts = scaled_prompts
            
        elif prompt_style == "point_prompt":
            # Scale center points
            scaled_prompts = []
            for point in prompts:
                x, y = point
                scaled_point = [
                    int(x * scale_x),
                    int(y * scale_y)
                ]
                scaled_prompts.append(scaled_point)
            prompts = scaled_prompts
    
    # 根据labels对prompts进行排序
    if len(prompts) > 0 and len(labels) > 0:
        # 创建索引数组，根据labels排序
        sorted_indices = sorted(range(len(labels)), key=lambda i: labels[i])
        
        # 重新排序prompts和labels
        prompts = [prompts[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
    
    return img, (prompts, labels)

def show_bboxes_labels_on_img(img, bboxes, labels):
    """
    Display bounding boxes and labels on the image.
    
    Args:
        image_path (str): Path to the image file.
        bboxes (list): List of bounding boxes in the format [x1, y1, x2, y2].
        labels (list): List of labels corresponding to the bounding boxes.
    """
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Image with Bounding Boxes', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def visualize_prompts_on_img(img, bboxes=None, masks=None, points=None, obj_ids=None, point_labels=None, alpha=0.5):
    """
    Visualize different types of prompts on the image.
    
    Args:
        img (np.ndarray): The input image.
        bboxes (list, optional): List of bounding boxes [x1, y1, x2, y2].
        masks (list, optional): List of binary masks.
        points (list, optional): List of points [x, y].
        obj_ids (list, optional): List of object IDs for color mapping.
        point_labels (list, optional): List of point labels (1 for positive, 0 for negative).
        alpha (float): Transparency factor for mask overlay.
        
    Returns:
        np.ndarray: Image with prompts visualized.
    """
    # Create a copy of the image
    prompt_img = img.copy()
    
    # Draw bounding boxes
    if bboxes is not None and len(bboxes) > 0:
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            obj_id = obj_ids[i] if obj_ids is not None and i < len(obj_ids) else i
            color = get_color_by_obj_id(obj_id)
            cv2.rectangle(prompt_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            cv2.putText(prompt_img, f'bbox_{obj_id}', (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Draw masks
    if masks is not None and len(masks) > 0:
        for i, mask in enumerate(masks):
            # Ensure mask is numpy array
            if not isinstance(mask, np.ndarray):
                mask = np.array(mask)
            
            obj_id = obj_ids[i] if obj_ids is not None and i < len(obj_ids) else i
            color = get_color_by_obj_id(obj_id)
            
            # 改进的mask渲染方法：只在mask区域进行混合，避免整体变暗
            mask_bool = mask > 0
            if np.any(mask_bool):
                # 在mask区域进行颜色混合
                for j in range(3):  # BGR channels
                    prompt_img[mask_bool, j] = (color[j] * alpha + 
                                              prompt_img[mask_bool, j] * (1 - alpha))
            
            # Find contours to place label
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.putText(prompt_img, f'mask_{obj_id}', (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Draw points
    if points is not None and len(points) > 0:
        for i, point in enumerate(points):
            x, y = int(point[0]), int(point[1])
            obj_id = obj_ids[i] if obj_ids is not None and i < len(obj_ids) else i
            
            # Determine color based on point label
            if point_labels is not None and i < len(point_labels):
                color = (0, 255, 0) if point_labels[i] > 0 else (0, 0, 255)
                label_text = 'pos' if point_labels[i] > 0 else 'neg'
            else:
                color = get_color_by_obj_id(obj_id)
                label_text = 'pos'
            
            # Draw circle for the point
            cv2.circle(prompt_img, (x, y), 8, color, -1)  # Filled circle
            cv2.circle(prompt_img, (x, y), 10, (255, 255, 255), 2)  # White border
            
            # Place label text
            text = f"pt_{obj_id}({label_text})"
            cv2.putText(prompt_img, text, (x + 15, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return prompt_img


def visualize_prediction_masks(img, masks, obj_ids=None, scores=None, alpha=0.3):
    """
    Visualize prediction masks on the image.
    
    Args:
        img (np.ndarray): The input image.
        masks (np.ndarray or list): Prediction masks.
        obj_ids (list, optional): List of object IDs for color mapping.
        scores (np.ndarray or list, optional): Confidence scores for each mask.
        alpha (float): Transparency factor for mask overlay.
        
    Returns:
        np.ndarray: Image with prediction masks visualized.
    """
    # Create a copy of the image
    pred_img = img.copy()
    
    if masks is not None and len(masks) > 0:
        for i, mask in enumerate(masks):
            obj_id = obj_ids[i] if obj_ids is not None and i < len(obj_ids) else i
            color = get_color_by_obj_id(obj_id)
            
            # 改进的mask渲染方法：只在mask区域进行混合，避免整体变暗
            mask_bool = mask > 0
            if np.any(mask_bool):
                # 在mask区域进行颜色混合
                for j in range(3):  # BGR channels
                    pred_img[mask_bool, j] = (color[j] * alpha + 
                                            pred_img[mask_bool, j] * (1 - alpha))
            
            # Find contours to place label
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Create label with score if available
                if scores is not None and i < len(scores):
                    label_text = f'pred_{obj_id}({scores[i]:.2f})'
                else:
                    label_text = f'pred_{obj_id}'
                
                cv2.putText(pred_img, label_text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return pred_img


def create_side_by_side_visualization(img, 
                                    bboxes=None, masks_prompt=None, points=None, obj_ids=None, point_labels=None,
                                    prediction_masks=None, prediction_obj_ids=None, prediction_scores=None,
                                    save_path=None, alpha=0.5):
    """
    Create side-by-side visualization: left side shows prompts, right side shows predictions.
    
    Args:
        img (np.ndarray): The input image.
        bboxes (list, optional): List of bounding boxes for prompts.
        masks_prompt (list, optional): List of mask prompts.
        points (list, optional): List of point prompts.
        obj_ids (list, optional): List of object IDs for prompts.
        point_labels (list, optional): List of point labels (1 for positive, 0 for negative).
        prediction_masks (np.ndarray or list, optional): Prediction masks.
        prediction_obj_ids (list, optional): List of object IDs for predictions (defaults to obj_ids if not provided).
        prediction_scores (np.ndarray or list, optional): Prediction scores.
        save_path (str, optional): Path to save the visualization.
        alpha (float): Transparency factor for overlays.
        
    Returns:
        np.ndarray: Side-by-side visualization image.
    """
    # Use obj_ids for predictions if prediction_obj_ids is not provided
    if prediction_obj_ids is None:
        prediction_obj_ids = obj_ids
    
    # Create prompt visualization (left side)
    prompt_img = visualize_prompts_on_img(img, bboxes=bboxes, masks=masks_prompt, 
                                        points=points, obj_ids=obj_ids, point_labels=point_labels, alpha=alpha)
    
    # Create prediction visualization (right side)
    pred_img = visualize_prediction_masks(img, prediction_masks, obj_ids=prediction_obj_ids, scores=prediction_scores, alpha=alpha)
    
    # Add titles
    prompt_img_with_title = prompt_img.copy()
    pred_img_with_title = pred_img.copy()
    
    # Add title text
    cv2.putText(prompt_img_with_title, 'Prompts', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(pred_img_with_title, 'Predictions', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Concatenate images horizontally
    combined_img = np.hstack([prompt_img_with_title, pred_img_with_title])
    
    # Save if path provided
    if save_path:
        cv2.imwrite(save_path, combined_img)
    
    return combined_img

def debug_annotation_scaling(json_path, image_path, image_size=1024):
    """
    Debug function to check annotation scaling issues.
    
    Args:
        json_path (str): Path to the Labelme JSON file.
        image_path (str): Path to the image file.
        image_size (int): Target size for resizing.
    """
    import json
    import chardet
    
    # Load original image
    original_img = cv2.imread(image_path)
    original_height, original_width = original_img.shape[:2]
    print(f"Original image size: {original_width} x {original_height}")
    
    # Load JSON
    with open(json_path, 'rb') as f:
        raw_data = f.read()
        encoding_result = chardet.detect(raw_data)
        detected_encoding = encoding_result['encoding'] or 'utf-8'
    
    with open(json_path, 'r', encoding=detected_encoding) as f:
        data = json.load(f)
    
    # Check annotation coordinates
    for i, shape in enumerate(data['shapes']):
        points = np.array(shape['points'], dtype=np.float32)
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
        print(f"Shape {i}: coords range X=[{min_x:.1f}, {max_x:.1f}], Y=[{min_y:.1f}, {max_y:.1f}]")
        
        if max_x > original_width or max_y > original_height:
            print(f"  WARNING: Coordinates exceed image bounds!")
    
    # Test the fixed function
    img, (prompts, labels) = get_imgs_labels_from_labelme_json(
        json_path, image_path, prompt_style="bbox_prompt", image_size=image_size
    )
    
    print(f"Processed image size: {img.shape[1]} x {img.shape[0]}")
    print(f"Number of prompts: {len(prompts)}")
    
    # Show scaled coordinates
    if len(prompts) > 0:
        for i, bbox in enumerate(prompts):
            x1, y1, x2, y2 = bbox
            print(f"Scaled bbox {i}: [{x1}, {y1}, {x2}, {y2}]")
            # Check if bboxes are reasonable size
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            bbox_area_ratio = (bbox_width * bbox_height) / (image_size * image_size)
            print(f"  Size: {bbox_width}x{bbox_height}, Area ratio: {bbox_area_ratio:.4f}")
    
    return img, prompts, labels


def create_prompt_results(image, bboxes=None, masks=None, obj_ids=None):
    """
    Create a Results object from prompt data for visualization
    
    Args:
        image (np.ndarray): Input image
        bboxes (list, optional): List of bounding boxes [x1, y1, x2, y2]
        masks (list, optional): List of binary masks
        obj_ids (list, optional): List of object IDs
    
    Returns:
        Results: Ultralytics Results object or None if no data
    """
    # If no data provided, return None
    if (bboxes is None or len(bboxes) == 0) and (masks is None or len(masks) == 0):
        return None
    
    # Get image shape
    orig_img = image
    img_shape = image.shape[:2]  # (H, W)
    
    # Prepare data tensors
    if bboxes is not None and len(bboxes) > 0:
        # Convert bboxes to tensor format [x1, y1, x2, y2, conf, cls]
        boxes_data = []
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            conf = 1.0  # Prompt confidence is 1.0
            cls = obj_ids[i] if obj_ids and i < len(obj_ids) else i
            boxes_data.append([x1, y1, x2, y2, conf, cls-1])
        
        boxes_tensor = torch.tensor(boxes_data, dtype=torch.float32)
    else:
        boxes_tensor = torch.empty((0, 6), dtype=torch.float32)
    
    if masks is not None and len(masks) > 0:
        # Convert masks to tensor format
        masks_tensor = torch.stack([torch.from_numpy(mask.astype(np.float32)) for mask in masks])
    else:
        masks_tensor = torch.empty((0, *img_shape), dtype=torch.float32)
    
    # Create Results object - 先用临时的names，后续会在可视化时统一
    results = Results(
        orig_img=orig_img,
        path="prompt_visualization",
        names={},  # 先设为空，后续统一设置
        boxes=boxes_tensor if len(boxes_tensor) > 0 else torch.empty((0, 6), dtype=torch.float32),
        masks=masks_tensor if len(masks_tensor) > 0 else torch.empty((0, *img_shape), dtype=torch.float32)
    )
    
    return results


def create_side_by_side_with_results(prompt_results, prediction_results, save_path=None):
    """
    Create side-by-side visualization using Results objects
    
    Args:
        prompt_results (Results): Results object for prompts (can be None for no prompts)
        prediction_results (Results): Results object for predictions
        save_path (str, optional): Path to save the visualization
    
    Returns:
        np.ndarray: Side-by-side visualization image
    """
    # Handle case where there are no prompts
    if prompt_results is None:
        # Create empty prompt image
        prompt_img = prediction_results.orig_img.copy()
        cv2.putText(prompt_img, 'No Prompts', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        # 确保两个Results对象使用相同的names字典来保证颜色一致性
        if prediction_results.names is not None:
            # 使用prediction的names字典
            unified_names = prediction_results.names.copy()
        else:
            # 创建统一的names字典
            unified_names = {}
        
        # 收集所有需要的class IDs
        all_cls = []
        if prompt_results.boxes is not None and len(prompt_results.boxes) > 0:
            prompt_cls = prompt_results.boxes.cls.cpu().numpy().astype(int).tolist()
            all_cls.extend(prompt_cls)
        if prediction_results.boxes is not None and len(prediction_results.boxes) > 0:
            pred_cls = prediction_results.boxes.cls.cpu().numpy().astype(int).tolist()
            all_cls.extend(pred_cls)
        
        # 确保unified_names包含所有需要的class IDs
        if all_cls:
            for cls_id in all_cls:
                if cls_id not in unified_names:
                    unified_names[cls_id] = str(cls_id)
        
        # 如果unified_names为空，创建默认的
        if not unified_names and all_cls:
            max_cls = max(all_cls)
            unified_names = {i: str(i) for i in range(max_cls + 1)}
        
        # 更新names字典以确保颜色一致
        prompt_results.names = unified_names
        prediction_results.names = unified_names
        
        # 添加调试信息
        print(f"Unified names dict: {unified_names}")
        if prompt_results.boxes is not None and len(prompt_results.boxes) > 0:
            prompt_cls = prompt_results.boxes.cls.cpu().numpy().astype(int).tolist()
            print(f"Prompt classes: {prompt_cls}")
        if prediction_results.boxes is not None and len(prediction_results.boxes) > 0:
            pred_cls = prediction_results.boxes.cls.cpu().numpy().astype(int).tolist()
            print(f"Prediction classes: {pred_cls}")
        
        # Use Ultralytics plot method for visualization with error handling
        try:
            prompt_img = prompt_results.plot(conf=False, line_width=3, font_size=12)
        except Exception as e:
            print(f"Warning: prompt plot failed ({e}), using original image")
            prompt_img = prompt_results.orig_img.copy()
    
    try:
        pred_img = prediction_results.plot(conf=True, line_width=3, font_size=12)
    except Exception as e:
        print(f"Warning: prediction plot failed ({e}), using original image")
        pred_img = prediction_results.orig_img.copy()
    
    # Add titles
    prompt_img_with_title = prompt_img.copy()
    pred_img_with_title = pred_img.copy()
    
    # Add title text
    cv2.putText(prompt_img_with_title, 'Prompts', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(pred_img_with_title, 'Predictions', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Concatenate images horizontally
    combined_img = np.hstack([prompt_img_with_title, pred_img_with_title])
    
    # Save if path provided
    if save_path:
        cv2.imwrite(save_path, combined_img)
    
    return combined_img

