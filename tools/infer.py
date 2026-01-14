
import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)




from ultralytics import YOLOE




def infer_yoloe_26s_seg():
    scale="26s"

    model=f"weights/yoloe-{scale}-seg.pt"

    model=YOLOE(model)
    model.args['clip_weight_name']="mobileclip2:b"

    # Set text prompt to detect person and bus. You only need to do this once after you load the model.
    names = ["person", "bus"]
    model.set_classes(names, model.get_text_pe(names))


    # infer test
    img_path="./ultralytics/assets/bus.jpg"
    results=model.predict(img_path,conf=0.25)
    results[0].save("runs/res-{}.jpg".format(scale))

def infer_yoloe26s_pf(end2end=True):


    yoloe26n_pf="runs/yoloe26_pf/26n_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra8]/weights/best.pt"
    yoloe26s_pf="runs/yoloe26_pf/26s_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra8]/weights/best.pt"
    yoloe26m_pf="runs/yoloe26_pf/26m_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra8]/weights/best.pt"
    yoloe26l_pf="runs/yoloe26_pf/26l_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra6]/weights/best.pt"
    yoloe26x_pf="runs/yoloe26_pf/26x_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra6]/weights/best.pt"

    scale="26x"
    model=f"weights/yoloe-{scale}-seg-pf.pt"

    model=YOLOE(model)
    model.args['clip_weight_name']="mobileclip2:b"

    if not end2end:
        model.model.end2end=False
        model.model.model[-1].end2end=False
    else:
        model.model.end2end=True
        model.model.model[-1].end2end=True


    # infer test
    img_path="./ultralytics/assets/bus.jpg"
    results=model.predict(img_path,conf=0.25)
    func_name="infer_yoloe26s_pf"
    results[0].save("runs/res-{func_name}.jpg".format(func_name=func_name))



infer_yoloe26s_pf(end2end=False)


def print_model():

    yoloe26n_pf="runs/yoloe26_pf/26n_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra8]/weights/best.pt"

    yoloe26n_tp="runs/yoloe26_tp/26n_ptwobjv1_bs256_epo30_close2_engine_old_engine_data_tp[ultra8]/weights/best.pt"

    model=YOLOE(yoloe26n_tp)
    model.args['clip_weight_name']="mobileclip2:b"
    model_head=model.model.model[-1]
    print(model_head.cv3[0])

# print_model()
