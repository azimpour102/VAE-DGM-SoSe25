python train.py \
--device cuda \
--dataset octmnist \
--batch_size 128 \
--epochs 5 \
--learning_rate 0.001 \
--model_name FullyConnectedVAE \
--model_path FullyConnectedVAE.pth \
--load_model True \
--save_model True