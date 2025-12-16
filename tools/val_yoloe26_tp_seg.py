import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)







path="./runs/yoloe26s_tp_seg_ultra6/mobileclip2:b_26s-seg_bs128_epo30_close2_opMuSGD_o2m0.1_segsegFalse_vpseg7/weights/best.pt"



from ultralytics import YOLOE

model=YOLOE(path)

data="../datasets/lvis.yaml"

metrics = model.val(data=data, split="val", max_det=1000,  save_json=True,task="segment")