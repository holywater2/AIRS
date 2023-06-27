conda activate potnet
# conda activate alignn-segnn
_MODEL_NUM=15
cd /home/holywater2/2023/AIRS/OpenMat/PotNet
CUDA_VISIBLE_DEVICES="6" CUBLAS_WORKSPACE_CONFIG=:16:8 python3 main.py --config /home/holywater2/2023/AIRS/OpenMat/PotNet/configs/potnet_$_MODEL_NUM.yaml --output_dir outputs/model_$_MODEL_NUM
cd /home/holywater2/2023/AIRS/OpenMat/PotNet/scripts
