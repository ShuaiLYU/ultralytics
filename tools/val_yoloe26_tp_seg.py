import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)





from ultralytics import YOLOE


# path="./runs/yoloe26s_tp_seg_ultra6/mobileclip2:b_26s-seg_bs128_epo30_close2_opMuSGD_o2m0.1_segsegFalse_vpseg7/weights/best.pt"

# model=YOLOE("yoloe-26s-seg.yaml").load("./weights/yoloe-26s.pt").to("cuda:0")


path="./runs/yoloe26s_tp_seg_ultra6/mobileclip2:b_26s-seg_bs128_epo30_close2_opMuSGD_o2m0.1_segsegFalse_segment26_mdata1_detachedproto_tp/weights/best.pt"

path="./runs/yoloe26s_tp_seg_ultra6/mobileclip2:b_26s-seg_bs128_epo30_close2_opMuSGD_o2m0.1_segment26_engine1_trainseghead_tp/weights/best.pt"



model=YOLOE("yoloe-26s-seg.yaml").load(path).to("cuda:3")






metrics = model.val(data="../datasets/lvis.yaml", split="val", max_det=1000, batch=1, save_json=True,task="segment")