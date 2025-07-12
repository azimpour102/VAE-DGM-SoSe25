python train.py \
--dataset bloodmnist,dermamnist,pathmnist \
--batch_size 128 \
--epochs 0 \
--learning_rate 0.001 \
--model_name ConditionalConvolutionalVAE \
--model_path bloodmnist_dermamnist_ConditionalConvolutionalVAE.pth \
--load_model
