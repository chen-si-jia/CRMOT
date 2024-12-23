cd src

# # In-domain inference：
# CUDA_VISIBLE_DEVICES=1 python track.py mot --load_model "/mnt/A/hust_csj/Code/Github/CRMOT/CRTracker/our_trained_models/CRTracker_model_20.pth" \
#                     --test_divo --track_display 1 --exp_name CRTracker_In-domain


# Cross-domain inference：
CUDA_VISIBLE_DEVICES=1 python track.py mot --load_model "/mnt/A/hust_csj/Code/Github/CRMOT/CRTracker/our_trained_models/CRTracker_model_20.pth" \
                    --test_campus --track_display 1 --exp_name CRTracker_Cross-domain


cd ..