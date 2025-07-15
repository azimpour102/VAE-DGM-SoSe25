#!/bin/bash

echo "Starting Enhanced VAE Training..."
echo "================================="

# Enhanced training with improved model
python train.py \
--dataset bloodmnist,dermamnist,pathmnist \
--batch_size 64 \
--epochs 30 \
--learning_rate 0.001 \
--model_name ImprovedConditionalConvolutionalVAE \
--model_path enhanced_conditional_model.pth

echo ""
echo "Training completed! Check the generated files:"
echo "- Performance analysis: *_performance_analysis.png"
echo "- Reconstructions: *_reconstruction.png" 
echo "- Generated samples: *_generation_*.png"
echo "- Latent analysis: *_latent_analysis.png"
echo "- Best model: best_enhanced_conditional_model.pth"
