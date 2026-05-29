


#  backgournd:




# train  yoloe26m on mergedata_v4


nohuppython train_yoloe26.py --model_version 26m --weight_path yoloe-26m-seg.pt --trainer YOLOETrainerFromScratch --data mergedata_v4 --optimizer MuSGD --lr0 0.00125 --lrf 0.5 --momentum 0.9 --weight_decay 0.0005 --epochs 50 --close_mosaic 1000 --scale 0.1 --batch 192 --device 0,1,2 --copy_paste 0.1 --mixup 0.0 --save_json True --project yoloa --name 26m_mergedata_v4


sync_best_pt /yoloa/26m_mergedata_v4/weights/best.pt 

python tools/build_yoloa.py --config ./tools/configs/yoloa_l6_320_26m_mergedata_v4.yaml --all mvtec && \
python tools/val_yoloa.py --config ./tools/configs/yoloa_l6_320_26m_mergedata_v4.yaml --all mvtec --mode yoloa
python tools/val_yoloa.py --config ./tools/configs/yoloa_l6_320_26m_mergedata_v4.yaml --all mvtec --mode yolo



  <!-- Anomaly val  (config=yoloa_l6_320_26m_mergedata_v4, mode=yolo)
========================================================================================================================
  name                           n_val    mAP10    mAP25    mAP50   mAP50-95        P        R  img_auroc  pix_auroc
  ------------------------------------------------------------------------------------------------------------------
  leather/yolo                     124   0.1602   0.1517   0.1033     0.0596   0.1746   0.1158        nan        nan
  grid/yolo                         78   0.1596   0.1493   0.1312     0.0680   0.2222   0.1429        nan        nan
  tile/yolo                        117   0.2246   0.0924   0.0475     0.0308   0.1229   0.6047        nan        nan
  wood/yolo                         79   0.3020   0.2774   0.1240     0.0638   0.3749   0.2987        nan        nan
  carpet/yolo                      117   0.1218   0.1104   0.1064     0.0782   0.4669   0.0745        nan        nan
  cable/yolo                       150   0.0562   0.0501   0.0294     0.0134   0.1054   0.3158        nan        nan
  hazelnut/yolo                    110   0.2436   0.1711   0.1143     0.0645   0.2797   0.2895        nan        nan
  pill/yolo                        167   0.0446   0.0399   0.0336     0.0194   0.0877   0.0892        nan        nan
  screw/yolo                       160   0.0257   0.0143   0.0066     0.0037   0.0366   0.1783        nan        nan
  toothbrush/yolo                   42   0.0955   0.0620   0.0289     0.0141   0.1798   0.1818        nan        nan
  metal_nut/yolo                   115   0.1025   0.0913   0.0804     0.0502   0.3109   0.1000        nan        nan
  capsule/yolo                     132   0.3570   0.3305   0.2503     0.0913   0.3871   0.4519        nan        nan
  bottle/yolo                       83   0.1609   0.0650   0.0484     0.0191   0.2791   0.2727        nan        nan
  transistor/yolo                  100   0.0000   0.0000   0.0000     0.0000   0.0000   0.0000        nan        nan
  zipper/yolo                      151   0.1197   0.0820   0.0509     0.0152   0.1493   0.1149        nan        nan
  ------------------------------------------------------------------------------------------------------------------
  AVERAGE                         1725   0.1449   0.1125   0.0770     0.0394   0.2118   0.2154        nan        nan -->



# train yolo26m on mergedata_v4_binary

nohupyolo train model=yolo26m.pt data=/home/louis/ultra_louis_work/datasets/AnomalyDataset/merge_data_v4_binary/data.yaml epochs=50 batch=96 close_mosaic=20 device=3,4,5 optimizer=MuSGD lr0=0.00125 lrf=0.5 momentum=0.9 weight_decay=0.0005 scale=0.1 copy_paste=0.1 mixup=0.0 save_json=True project=yoloa name=26m_yolo_v4_binary_cm20_v1


sync_best_pt /yoloa/26m_yolo_v4_binary_cm20_v1/weights/best.pt 

python tools/build_yoloa.py --config ./tools/configs/yoloa_l6_320_26m_yolo_v4_binary_cm20_v1.yaml --all mvtec && \
python tools/val_yoloa.py --config ./tools/configs/yoloa_l6_320_26m_yolo_v4_binary_cm20_v1.yaml --all mvtec --mode yoloa

<!-- ========================================================================================================================
  Anomaly val  (config=yoloa_l6_320_26m_yolo_v4_binary_cm20_v1, mode=yoloa)
========================================================================================================================
  name                           n_val    mAP10    mAP25    mAP50   mAP50-95        P        R  img_auroc  pix_auroc
  ------------------------------------------------------------------------------------------------------------------
  leather/yoloa                    124   0.7043   0.5599   0.3146     0.1058   0.7284   0.6211     0.9823     0.9879
  grid/yoloa                        78   0.5667   0.4278   0.1121     0.0450   0.9130   0.2143     0.9683     0.9596
  tile/yoloa                       117   0.6833   0.4961   0.3018     0.1135   0.6456   0.5930     0.9921     0.8884
  wood/yoloa                        79   0.7513   0.6887   0.3715     0.1466   0.9333   0.5455     0.9588     0.9368
  carpet/yoloa                     117   0.9202   0.8140   0.4904     0.2174   0.9639   0.8511     0.9872     0.9828
  cable/yoloa                      150   0.4225   0.3468   0.2071     0.0967   0.5059   0.3772     0.9363     0.9078
  hazelnut/yoloa                   110   0.4262   0.3281   0.1435     0.0703   0.4853   0.4342     0.9993     0.9731
  pill/yoloa                       167   0.4432   0.4075   0.2887     0.1261   0.4715   0.3494     0.9599     0.9583
  screw/yoloa                      160   0.1708   0.1444   0.1136     0.0535   0.3000   0.0465     0.9053     0.9808
  toothbrush/yoloa                  42   0.0583   0.0360   0.0172     0.0043   0.1000   0.0909     0.9000     0.9848
  metal_nut/yoloa                  115   0.1508   0.1091   0.0264     0.0116   0.2170   0.2091     0.9956     0.9173
  capsule/yoloa                    132   0.4793   0.3316   0.2194     0.1122   0.5161   0.4404     0.9852     0.9858
  bottle/yoloa                      83   0.3532   0.2319   0.1470     0.0547   0.3639   0.3788     0.9992     0.9792
  transistor/yoloa                 100   0.1829   0.1829   0.0000     0.0000   0.2500   0.0930     0.9950     0.7844
  zipper/yoloa                     151   0.4808   0.1737   0.0634     0.0145   0.5398   0.3506     0.9850     0.9641
  ------------------------------------------------------------------------------------------------------------------
  AVERAGE                         1725   0.4529   0.3519   0.1878     0.0781   0.5289   0.3730     0.9700     0.9461 -->

python tools/val_yoloa.py --config ./tools/configs/yoloa_l6_320_26m_yolo_v4_binary_cm20_v1.yaml --all mvtec --mode yolo

<!-- ========================================================================================================================
  name                           n_val    mAP10    mAP25    mAP50   mAP50-95        P        R  img_auroc  pix_auroc
  ------------------------------------------------------------------------------------------------------------------
  leather/yolo                     124   0.4267   0.3988   0.3101     0.1682   0.2763   0.4421        nan        nan
  grid/yolo                         78   0.2006   0.1515   0.0751     0.0390   0.1752   0.2449        nan        nan
  tile/yolo                        117   0.4323   0.3312   0.2479     0.1615   0.6589   0.2695        nan        nan
  wood/yolo                         79   0.6948   0.6573   0.5886     0.2923   0.6713   0.6234        nan        nan
  carpet/yolo                      117   0.6566   0.6075   0.4724     0.3017   0.8285   0.5139        nan        nan
  cable/yolo                       150   0.1019   0.0943   0.0513     0.0295   0.1460   0.4561        nan        nan
  hazelnut/yolo                    110   0.5657   0.4934   0.3579     0.2700   0.5646   0.5263        nan        nan
  pill/yolo                        167   0.0716   0.0662   0.0607     0.0328   0.0888   0.1084        nan        nan
  screw/yolo                       160   0.0649   0.0481   0.0341     0.0211   0.0771   0.2248        nan        nan
  toothbrush/yolo                   42   0.2073   0.1541   0.0757     0.0513   0.3935   0.2576        nan        nan
  metal_nut/yolo                   115   0.1209   0.1084   0.0714     0.0515   0.1417   0.1545        nan        nan
  capsule/yolo                     132   0.5832   0.5125   0.3740     0.1537   0.7573   0.5229        nan        nan
  bottle/yolo                       83   0.2561   0.1700   0.1052     0.0546   0.3760   0.2727        nan        nan
  transistor/yolo                  100   0.0112   0.0020   0.0015     0.0005   0.1107   0.0233        nan        nan
  zipper/yolo                      151   0.1358   0.0908   0.0371     0.0125   0.1312   0.1667        nan        nan
  ------------------------------------------------------------------------------------------------------------------
  AVERAGE                         1725   0.3020   0.2591   0.1909     0.1093   0.3598   0.3205        nan        nan -->

<!-- 

# set yolo_weight=0.5 in head.py and re-run yoloa val
========================================================================================================================
  Anomaly val  (config=yoloa_l6_320_26m_yolo_v4_binary_cm20_v1, mode=yoloa)
========================================================================================================================
  name                           n_val    mAP10    mAP25    mAP50   mAP50-95        P        R  img_auroc  pix_auroc
  ------------------------------------------------------------------------------------------------------------------
  leather/yoloa                    124   0.4754   0.4754   0.4121     0.1772   0.8750   0.0737     0.9823     0.9879
  grid/yoloa                        78   0.5102   0.5102   0.5102     0.3046   1.0000   0.0204     0.9683     0.9596
  tile/yoloa                       117   0.5151   0.5151   0.3123     0.2102   0.9091   0.1163     0.9921     0.8884
  wood/yoloa                        79   0.5519   0.5519   0.3878     0.1570   1.0000   0.1039     0.9588     0.9368
  carpet/yoloa                     117   0.5479   0.5479   0.5479     0.3346   1.0000   0.0957     0.9872     0.9828
  cable/yoloa                      150   0.4381   0.4381   0.3805     0.2528   0.7500   0.1316     0.9363     0.9078
  hazelnut/yoloa                   110   0.5842   0.5131   0.4456     0.3157   0.9444   0.2237     0.9993     0.9731
  pill/yoloa                       167   0.3690   0.3479   0.2984     0.2085   0.5682   0.1506     0.9599     0.9583
  screw/yoloa                      160   0.0000   0.0000   0.0000     0.0000   0.0000   0.0000     0.9053     0.9808
  toothbrush/yoloa                  42   0.1550   0.1550   0.0766     0.0077   0.5864   0.0303     0.9000     0.9848
  metal_nut/yoloa                  115   0.1547   0.1547   0.1075     0.0657   0.2500   0.0727     0.9956     0.9173
  capsule/yoloa                    132   0.4377   0.4077   0.3015     0.1593   0.6667   0.2018     0.9852     0.9858
  bottle/yoloa                      83   0.5276   0.3597   0.2274     0.1369   0.8571   0.1818     0.9992     0.9792
  transistor/yoloa                 100   0.1747   0.0000   0.0000     0.0000   0.3333   0.0233     0.9950     0.7844
  zipper/yoloa                     151   0.4142   0.2113   0.1046     0.0209   0.8000   0.0230     0.9850     0.9641
  ------------------------------------------------------------------------------------------------------------------
  AVERAGE                         1725   0.3904   0.3459   0.2741     0.1567   0.7027   0.0966     0.9700     0.9461 
  
  seems: setting yolo_weight=0.5 significantly improves precision but hurts recall, which indicates the blended yolo score is more confident but less proposals are generated. We may need to adjust the confidence threshold for proposal generation to improve recall.
  -->
python tools/predict_yoloa.py --config tools/configs/yoloa_l6_320_26m_yolo_v4_binary_cm20_v1.yaml --all mvtec --kind any



# change to yoloe26l

nohuppython train_yoloe26.py --model_version 26l --weight_path yoloe-26l-seg.pt --trainer YOLOETrainerFromScratch --data mergedata_v4 --optimizer MuSGD --lr0 0.00125 --lrf 0.5 --momentum 0.9 --weight_decay 0.0005 --epochs 50 --close_mosaic 1000 --scale 0.1 --batch 256 --device 0,1,2,3 --copy_paste 0.1 --mixup 0.0 --save_json True --project yoloa --name 26l_mergedata_v4


