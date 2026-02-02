from ultralytics import YOLOE,YOLO


import os


import argparse
argparser=argparse.ArgumentParser(description="merge visual prompt and seg head for yoloe")
argparser.add_argument("--scale", type=str, default="26s-seg")
argparser.add_argument("--vp_weight", type=str, default="yoloe26s-vp.pt")
argparser.add_argument("--seg_weight", type=str, default="yoloe-26s-seg-det.pt")
argparser.add_argument("--im", type=str, default="ultralytics/assets/bus.jpg")
argparser.add_argument("--save_dir", type=str, default="runs/fuse_seg_vp")
args=argparser.parse_args() 

scale=args.scale
vp_weight=args.vp_weight
seg_weight=args.seg_weight
im=args.im
save_dir=args.save_dir  

os.makedirs(save_dir,exist_ok=True)


model=YOLO(f"yoloe-{scale}.yaml")

# fafa
model.load(seg_weight)
model.args['clip_weight_name']="mobileclip2:b"
model.load(vp_weight)

model.save(f"weights/yoloe-{scale}-ye.pt")



names = ["person", "bus"]
model.set_classes(names, model.get_text_pe(names))
# 
results_seg=model.predict(source=im,conf=0.25,task="segment",save=False)
results_seg[0].save(filename=f"{save_dir}/segmentation_result.png")


import numpy as np

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# Define visual prompts using bounding boxes and their corresponding class IDs.
# Each box highlights an example of the object you want the model to detect.
visual_prompts = dict(
    bboxes=np.array(
        [
            [221.52, 405.8, 344.98, 857.54],  # Box enclosing person
            [120, 425, 160, 445],  # Box enclosing glasses
        ],
    ),
    cls=np.array(
        [
            0,  # ID to be assigned for person
            1,  # ID to be assigned for glasses
        ]
    ),
)



# Run inference on an image, using the provided visual prompts as guidance
results = model.predict(
    im,
    visual_prompts=visual_prompts,
    predictor=YOLOEVPSegPredictor,
    conf=0.01,
)

# Show results
import os 



results[0].save(filename=f"{save_dir}/vp_segmentation_result.png")
print("Results saved to:", f"{os.path.abspath(save_dir)}/vp_segmentation_result.png")







# # 26s seg+vp fusion infer test
#  python tools/fuse_seg_vp.py --scale 26s-seg \
#  --seg_weight /home/louis/ultra_louis_work/ultralytics/weights/yoloe26ye/yoloe26_seg/26s-seg_ptwbest_tp_bs256_epo10_close2_engine_yoloe_ye_data.yaml_seg-ultra6/weights/best.pt \
#   --vp_weight /home/louis/ultra_louis_work/ultralytics/weights/yoloe26ye/yoloe26_vp/26s_ptwbest_tp_bs256_epo10_close2_engine_yedata_vp_ye\[ultra6\]/weights/best.pt \
#  --im ultralytics/assets/bus.jpg \
#  --save_dir runs/merge_vp_seg/

#  # 26n seg+vp fusion infer test
#     python tools/fuse_seg_vp.py --scale 26n-seg \
#     --seg_weight /home/louis/ultra_louis_work/ultralytics/weights/yoloe26ye/yoloe26_seg/26n-seg_ptwbest_tp_bs256_epo10_close2_engine_yoloe_ye_data.yaml_seg-ultra6/weights/best.pt \
#      --vp_weight /home/louis/ultra_louis_work/ultralytics/weights/yoloe26ye/yoloe26_vp/26n_ptwbest_tp_bs256_epo10_close2_engine_yedata_vp_ye\[ultra6\]/weights/best.pt \
#     --im ultralytics/assets/bus.jpg \
#      --save_dir runs/merge_vp_seg/


# # 26m seg+vp fusion infer test
#  python tools/fuse_seg_vp.py --scale 26m-seg \
#  --seg_weight /home/louis/ultra_louis_work/ultralytics/weights/yoloe26ye/yoloe26_seg/26m-seg_ptwbest_tp_bs256_epo10_close2_engine_ye_data_seg-ultra7/weights/best.pt \
#   --vp_weight /home/louis/ultra_louis_work/ultralytics/weights/yoloe26ye/yoloe26_vp/26m_ptwbest_tp_bs256_epo10_close2_engine_yedata_vp_ye\[ultra7\]/weights/best.pt \
#  --im ultralytics/assets/bus.jpg \
#  --save_dir runs/merge_vp_seg/

#  # 26l seg+vp fusion infer test
#     python tools/fuse_seg_vp.py --scale 26l-seg \
#     --seg_weight /home/louis/ultra_louis_work/ultralytics/weights/yoloe26ye/yoloe26_seg/26l-seg_ptwbest_tp_bs256_epo10_close2_engine_yoloe_ye_data.yaml_seg-ultra6/weights/best.pt \
#      --vp_weight /home/louis/ultra_louis_work/ultralytics/weights/yoloe26ye/yoloe26_vp/26l_ptwbest_tp_bs256_epo10_close2_engine_yedata_vp_ye\[ultra6\]/weights/best.pt \
#     --im ultralytics/assets/bus.jpg \
#      --save_dir runs/merge_vp_seg/


# # 26x seg+vp fusion infer test
#  python tools/fuse_seg_vp.py --scale 26x-seg \
#  --seg_weight /home/louis/ultra_louis_work/ultralytics/weights/yoloe26ye/yoloe26_seg/26x-seg_ptwbest_tp_bs256_epo10_close2_engine_yoloe_ye_data.yaml_seg-ultra4/weights/best.pt \
#   --vp_weight /home/louis/ultra_louis_work/ultralytics/weights/yoloe26ye/yoloe26_vp/26x_ptwbest_tp_bs256_epo10_close2_engine_yedata_vp_ye\[ultra7\]/weights/best.pt \
#  --im ultralytics/assets/bus.jpg \
#  --save_dir runs/merge_vp_seg/  

