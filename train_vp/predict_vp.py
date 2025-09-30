
import os ,ultralytics
from ultralytics.engine import results
os.chdir(os.path.dirname(os.path.dirname(ultralytics.__file__)))


import numpy as np

from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor,YOLOEVPDetectPredictor



# Define visual prompts using bounding boxes and their corresponding class IDs.
# Each box highlights an example of the object you want the model to detect.
# visual_prompts = dict(
#     bboxes=np.array(
#         [
#             [221.52, 405.8, 344.98, 857.54],  # Box enclosing person
#             [120, 425, 160, 445],  # Box enclosing glasses
#         ],
#     ),
#     cls=np.array(
#         [
#             0,  # ID to be assigned for person
#             1,  # ID to be assigned for glassses
#         ]
#     ),
# )

# # Run inference on an image, using the provided visual prompts as guidance
# results = model.predict(
#     "ultralytics/assets/bus.jpg",
#     visual_prompts=visual_prompts,
#     predictor=YOLOEVPDetectPredictor,
# )

# # Show results
# results[0].save(filename="yoloe_vp_detect.jpg")



##########################################################################################
# from ultralytics import YOLOE

# # Initialize a YOLOE model
# model_tp = YOLOE("yoloe-v8l-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes

# # Set text prompt to detect person and bus. You only need to do this once after you load the model.
# names = ["tie"]
# model_tp.set_classes(names, model_tp.get_text_pe(names))

# # Run detection on the given image
# results = model_tp.predict("ultralytics/assets/zidane.jpg",)

# # Show results
# print(results[0].boxes)

# results[0].save(filename="tie.jpg")

##############################################################################################################################
# def draw_bbox_on_img(img_path,bboxes,save_path=None):
#     import cv2
#     img = cv2.imread(img_path)
#     for box in bboxes:
#         x1,y1,x2,y2=box
#         cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
#     if save_path:
#         cv2.imwrite(save_path, img)
#     return img


# draw_bbox_on_img("ultralytics/assets/zidane.jpg",[[ 437.3754,  437.1382,  523.6042,  717.8223]],"zidane_box.jpg")


##############################################################################################################################


# Initialize a YOLOE model
model = YOLOE("yoloe-v8l-seg.pt")

results1 = model.predict(
    "ultralytics/assets/bus.jpg",
    visual_prompts = dict(bboxes=np.array([
        [221.52, 405.8, 344.98, 857.54],
          [120, 425, 160, 445],  # Box enclosing glasses
    ]),
    cls=["person","eyeglasses"]),
    predictor=YOLOEVPDetectPredictor,
)


results2 = model.predict(
    "ultralytics/assets/zidane.jpg",
    visual_prompts = dict(bboxes=np.array([[437.3754,  437.1382,  523.6042,  717.8223]]),
    cls=["tie"]),
    # cls=np.array([1])),
    # conf=0.0,
)


results3= model.predict(
    "ultralytics/assets/threepeople.jpg",
    # visual_prompts = dict(bboxes=np.array([[2.5918e+02, 3.1178e+01, 5.1011e+02, 4.2105e+02]]),
# cls=["person"]),
    conf=0.05,
)



results1[0].save(filename="yoloe_vp_detect1.jpg")
results2[0].save(filename="yoloe_vp_detect2.jpg")
results3[0].save(filename="yoloe_vp_detect3.jpg")


img1 = results1[0].plot()
img2 = results2[0].plot()
img3 = results3[0].plot()

# lettoxbox to 640x640

from ultralytics.data.augment import LetterBox
import numpy as np

# Create the LetterBox transform
letterbox = LetterBox(new_shape=(640, 480))

# Apply letterbox to get the padded image
img1 = letterbox(image=img1)
img2 = letterbox(image=img2)
img3 = letterbox(image=img3)

# concat the results and save to disk

import cv2
import numpy as np
final_img = np.concatenate((img1, img2, img3), axis=1)
cv2.imwrite("yoloe_vp_detect_combined.jpg", final_img)
