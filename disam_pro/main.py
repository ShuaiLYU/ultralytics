
import sys,os
from tkinter.tix import Tree
import cv2
import torch
import numpy as np
from  disam_utils import *
from ultralytics.models.sam import SAM2DynamicInteractivePredictor


'''
This is a demo script for using the SAM2DynamicInteractivePredictor with images and labels from Labelme JSON files.
It demonstrates how to load images, extract bounding boxes and labels from Labelme JSON files, and use the predictor to segment objects in the images. The results are displayed with bounding boxes and labels on the images.

'''

# set the current path as workspace
os.chdir( os.path.dirname( os.path.dirname( os.path.abspath(__file__) )  ) )


# prompt_type="bbox"  # or "mask"
# img_folder="user_img"
# def use_prompt():
#     return True if index in [ 0] else False


# for car_race_images


# prompt_type="bbox"  # or "mask"
# img_folder="car_race_images"
# def use_prompt():
#     return True if index in [0, 3] else False


# prompt_type="bbox"  # or "mask"
# img_folder="metal_images"
# def use_prompt():
#     return True if index in [0,3] else False

prompt_type="bbox"  # or "mask"
img_folder="mvtec_loco_juice_bottle"
def use_prompt():
    return True if index in [0] else False




if __name__ == "__main__":


    assert prompt_type in ["bbox", "mask"], "prompt_type must be either 'bbox' or 'mask' "
    # whether to use point prompt to improve the segmentation results
    use_point_prompt=True  


    print(f"Running demo with prompt_type: {prompt_type}")
    
    # Create SAM2DynamicInteractivePredictor
    overrides = dict(conf=0.01, task="segment", mode="predict", imgsz=1024, model="sam2_t.pt",save=False)
    predictor = SAM2DynamicInteractivePredictor(overrides=overrides,max_obj_num=6)


    # Load images and labels from Labelme JSON files
    image_dir= os.path.join(os.path.dirname(__file__),img_folder    )
    image_names= [ i for i in os.listdir(image_dir) if i.endswith('.jpg') or i.endswith('.png') or i.endswith('.bmp') ]
    res_dir= os.path.join(os.path.dirname(__file__), f'{img_folder}_results'    )
    os.makedirs(res_dir, exist_ok=True)


    # Iterate through images and their corresponding Labelme JSON files
    for index, image_name in enumerate(image_names):
        
        image_path= os.path.join(image_dir, image_name)


        #image_size=overrides["imgsz"]

            # return index < 3  # Use prompt for the first three images

        # For the first two images, use the provided bounding boxes and labels
        if use_prompt():
            
            # Find the corresponding JSON file
            # Try different extensions for the JSON file
            base_name = os.path.splitext(image_name)[0]  # Remove extension
            labelme_json_path = os.path.join(image_dir, f"{base_name}.json")
            
            # Check if JSON file exists
            if not os.path.exists(labelme_json_path):
                print(f"Warning: JSON file not found for {image_name}, skipping prompts for this image")
                # Use image without prompts
                image = read_img(image_path)
                results = predictor(source=image)
            else:
                if prompt_type == "bbox":
                    # check the labels, they should be in [1, max_obj_num]
                    image,(bboxes,obj_ids)=get_imgs_labels_from_labelme_json(
                        labelme_json_path, image_path,  prompt_style="bbox_prompt")
                    
                    # in addition, we can input points and point labels if available
                    _,(points,point_labels)=get_imgs_labels_from_labelme_json(
                        labelme_json_path, image_path,  prompt_style="point_prompt")
                    if not len(point_labels)>0 or (not use_point_prompt):
                        point_labels=None
                        points=None
                    results= predictor(source=image, bboxes=bboxes,obj_ids=obj_ids,points=points,labels=point_labels, update_memory=True)

                if prompt_type == "mask":
                    # check the labels, they should be in [1, max_obj_num]
                    image,(masks,obj_ids)=get_imgs_labels_from_labelme_json(
                        labelme_json_path, image_path,  prompt_style="mask_prompt")
                    print( "length of masks:", len(masks) )
                    print( "length of obj_ids:", len(obj_ids) )
            
                    # in addition, we can input points and point labels if available
                    _,(points,point_labels)=get_imgs_labels_from_labelme_json(
                        labelme_json_path, image_path,  prompt_style="point_prompt")
                    if not len(point_labels)>0 or (not use_point_prompt):
                        point_labels=None
                        points=None
                    results= predictor(source=image, masks=masks,obj_ids=obj_ids,points=points,labels=point_labels, update_memory=True)



            # 创建左右对比的可视化结果
            if prompt_type == "bbox":
                # 创建prompt的Results对象
                prompt_results = create_prompt_results(
                    image=image,
                    bboxes=bboxes,
                    obj_ids=obj_ids
                )
                
                # 使用Results对象进行可视化
                side_by_side_image = create_side_by_side_with_results(
                    prompt_results=prompt_results,
                    prediction_results=results[0]
                )
                
            else:  # mask
                # 创建prompt的Results对象
                prompt_results = create_prompt_results(
                    image=image,
                    masks=masks,
                    obj_ids=obj_ids
                )
                
                # 使用Results对象进行可视化
                side_by_side_image = create_side_by_side_with_results(
                    prompt_results=prompt_results,
                    prediction_results=results[0]
                )
                
            # 保存左右对比的结果
            save_path = os.path.join(res_dir, image_name.replace('.jpg', '_res.jpg'))
            cv2.imwrite(save_path, side_by_side_image)
            print(f"Side-by-side result saved to: {save_path}")

        # elif index == 5:
        #     # this is a simulation that the user can add new support image anytime
        #     results= predictor(source=image, bboxes=bboxes,obj_ids=labels,update_memory=True)
        else:

            # image,(bboxes,obj_ids)=get_imgs_labels_from_labelme_json(
            #     labelme_json_path, image_path,  prompt_style="bbox_prompt")
                    

            image=read_img(image_path)
            # For subsequent images, use the memory from the previous images
            results= predictor(source=image)
            
            # 创建只显示预测结果的可视化（没有prompt）
            # 使用Results对象进行可视化
            side_by_side_image = create_side_by_side_with_results(
                prompt_results=None,  # 没有prompt
                prediction_results=results[0]
            )
            
            # 保存结果
            save_path = os.path.join(res_dir, image_name.replace('.jpg', '_res.jpg'))
            cv2.imwrite(save_path, side_by_side_image)
            print(f"Side-by-side result saved to: {save_path}")
            
            # 输出预测信息
            if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                if hasattr(results[0].boxes, 'cls'):
                    prediction_obj_ids = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
                    print(f"Predicted object IDs: {prediction_obj_ids}")

 
