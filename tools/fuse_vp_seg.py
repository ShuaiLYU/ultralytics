import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)



from ultralytics import YOLOE
from copy import deepcopy



def merge_savpe_to_seg_model(scale, vp_weight,seg_weight, output_weight):
    """
    Merge the savpe module from the visual prompt model into the segmentation model.
    Args:
        scale: str, model scale, e.g., "26", "26s", "26m"
        vp_weight: str, path to the visual prompt model weight
        seg_weight: str, path to the segmentation model weight
        output_weight: str, path to save the merged model weight
    """
    seg_model = YOLOE(seg_weight)
    vp_model = YOLOE(vp_weight)

    seg_model.model.model[-1].savpe = deepcopy(vp_model.model.model[-1].savpe)
    seg_model.eval()
    seg_model.save(output_weight)


if __name__ == "__main__":



    yoloe26_vp="runs/yoloe26_vp/26n_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra8]/weights/best.pt"
    yoloe26s_vp="runs/yoloe26_vp/26s_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra8]/weights/best.pt"
    yoloe26m_vp="runs/yoloe26_vp/26m_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra8]/weights/best.pt"
    yoloe26l_vp="runs/yoloe26_vp/26l_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra6]/weights/best.pt"
    yoloe26x_vp="runs/yoloe26_vp/26x_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra6]/weights/best.pt"
         
    yoloe26n_seg="runs/yoloe26_seg/26n-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra2]/weights/best.pt"
    yoloe26s_seg="runs/yoloe26_seg/26s-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra8]/weights/best.pt"
    yoloe26m_seg="runs/yoloe26_seg/26m-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra2]/weights/best.pt"
    yoloe26l_seg="runs/yoloe26_seg/26l-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra6]/weights/best.pt"
    yoloe26x_seg="runs/yoloe26_seg/26x-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra6]/weights/best.pt"

    merge_savpe_to_seg_model("26n", yoloe26_vp, yoloe26n_seg, "./weights/yoloe-26n-seg.pt")
    # merge_savpe_to_seg_model("26s", yoloe26s_vp, yoloe26s_seg, "./weights/yoloe-26s-seg.pt")
    # merge_savpe_to_seg_model("26m", yoloe26m_vp, yoloe26m_seg, "./weights/yoloe-26m-seg.pt")
    # merge_savpe_to_seg_model("26l", yoloe26l_vp, yoloe26l_seg, "./weights/yoloe-26l-seg.pt")
    # merge_savpe_to_seg_model("26x", yoloe26x_vp, yoloe26x_seg, "./weights/yoloe-26x-seg.pt")