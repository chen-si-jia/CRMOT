cd src

# CRTracker 单视图推理
# CRTracker for single view inference：
CUDA_VISIBLE_DEVICES=1 python track_single_view_demo.py mot \
                    --load_model "/mnt/A/hust_csj/Code/CRMOT/CRTracker/models/CRTracker_model_20.pth" \
                    --track_display 0 \
                    --conf_thres 0.5

cd ..