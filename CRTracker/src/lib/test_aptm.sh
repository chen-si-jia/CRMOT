
# test APTM：
python3 aptm/inference_single_text_module.py --task "cuhk" --checkpoint "/mnt/A/hust_csj/Code/Github/CRMOT/CRTracker/models/APTM_models/checkpoints/ft_cuhk/checkpoint_best.pth" --config '/mnt/A/hust_csj/Code/Github/CRMOT/CRTracker/models/APTM_models/configs/Retrieval_cuhk.yaml'
python3 aptm/inference_single_text_module.py --task "icfg" --checkpoint "/mnt/A/hust_csj/Code/Github/CRMOT/CRTracker/models/APTM_models/checkpoints/ft_icfg/checkpoint_best.pth" --config '/mnt/A/hust_csj/Code/Github/CRMOT/CRTracker/models/APTM_models/configs/Retrieval_icfg.yaml'
python3 aptm/inference_single_text_module.py --task "pa100k" --checkpoint "/mnt/A/hust_csj/Code/Github/CRMOT/CRTracker/models/APTM_models/checkpoints/ft_pa100k/checkpoint_best.pth" --config '/mnt/A/hust_csj/Code/Github/CRMOT/CRTracker/models/APTM_models/configs/Retrieval_pa100k.yaml'
python3 aptm/inference_single_text_module.py --task "rstp" --checkpoint "/mnt/A/hust_csj/Code/Github/CRMOT/CRTracker/models/APTM_models/checkpoints/ft_rstp/checkpoint_best.pth" --config '/mnt/A/hust_csj/Code/Github/CRMOT/CRTracker/models/APTM_models/configs/Retrieval_rstp.yaml'
