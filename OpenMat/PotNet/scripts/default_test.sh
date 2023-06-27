conda activate potnet
# conda activate alignn-segnn
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/compuworks/anaconda3/lib
_MODEL_NUM=10
cd /home/holywater2/2023/AIRS/OpenMat/PotNet
CUDA_VISIBLE_DEVICES="0" CUBLAS_WORKSPACE_CONFIG=:16:8 python3 main.py --config /home/holywater2/2023/AIRS/OpenMat/PotNet/configs/potnet_test.yaml --output_dir outputs/model_test
cd /home/holywater2/2023/AIRS/OpenMat/PotNet/scripts
