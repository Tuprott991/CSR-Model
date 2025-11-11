# Quick Start Guide

This guide will help you get started with CSR model in under 10 minutes.

## 1. Installation (2 minutes)

```bash
# Clone repository
cd CSR-Model

# Install dependencies
pip install -r requirements.txt
```

## 2. Prepare Your Data (3 minutes)

### Option A: Use TBX11K format

Create a JSON file with annotations:

```json
{
  "image_001.png": {
    "findings": ["infiltration", "effusion"],
    "disease": "tuberculosis"
  },
  "image_002.png": {
    "findings": ["nodule"],
    "disease": "normal"
  }
}
```

### Option B: Use CSV format (VinDr-CXR / ISIC)

Create a CSV with columns:
- `image_id`: filename
- Binary columns for each concept (0/1)
- `disease`: disease label

```csv
image_id,infiltration,effusion,nodule,disease
img1.jpg,1,1,0,tuberculosis
img2.jpg,0,0,1,normal
```

### Directory structure:

```
data/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── train_annotations.json  # or .csv
├── val_annotations.json
└── test_annotations.json
```

## 3. Train the Model (5+ minutes)

### Quick training (all stages):

```bash
python train.py \
    --dataset tbx11k \
    --data_root ./data/images \
    --train_file ./data/train_annotations.json \
    --val_file ./data/val_annotations.json \
    --test_file ./data/test_annotations.json \
    --backbone resnet50 \
    --num_prototypes 10 \
    --batch_size 32 \
    --stage_a_epochs 30 \
    --stage_b_epochs 20 \
    --stage_c_epochs 30 \
    --save_dir checkpoints/my_model \
    --device cuda
```

**Optional: Use ConvNeXtV2 for better accuracy**:
```bash
# First install timm
pip install timm

# Then train with ConvNeXtV2
python train.py --backbone convnextv2_base [other args...]
```

### Monitor training:

The model will automatically save checkpoints:
- `stage_a_best.pth` - Best concept model
- `stage_b_best.pth` - Best prototypes
- `stage_c_best.pth` - Best final model (use this for inference!)

## 4. Evaluate the Model

```bash
python evaluate.py \
    --dataset tbx11k \
    --data_root ./data/images \
    --train_file ./data/train_annotations.json \
    --val_file ./data/val_annotations.json \
    --test_file ./data/test_annotations.json \
    --checkpoint checkpoints/my_model/stage_c_best.pth \
    --backbone resnet50 \
    --num_prototypes 10 \
    --output_dir results/
```

Results will be saved in `results/evaluation_report.json`

## 5. Try the Demos

```bash
python demo.py
```

This will show you:
- Basic inference
- Concept rejection
- Spatial feedback
- Interactive diagnosis
- Visualizations

## 6. Use in Your Code

```python
from model import CSRModel
from doctor_interaction import DoctorInteraction
import torch

# Load trained model
model = CSRModel(
    num_concepts=15,  # Adjust to your dataset
    num_classes=3,    # Adjust to your dataset
    num_prototypes_per_concept=10
)

checkpoint = torch.load('checkpoints/my_model/stage_c_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make prediction
image = torch.randn(1, 3, 224, 224)  # Replace with your image
outputs = model(image, stage='task')
prediction = outputs['task_logits'].argmax(dim=1)

print(f"Predicted class: {prediction.item()}")

# Use doctor interaction
doctor = DoctorInteraction(model)
result = doctor.interactive_diagnosis(image)
print(f"Top concepts: {result['top_concepts']}")
```

## Common Issues

### Issue: CUDA out of memory
**Solution**: Reduce batch size or use CPU (`--device cpu`)

### Issue: Poor accuracy
**Solutions**:
- Train longer (increase epochs)
- Use data augmentation (already included in dataloader)
- Adjust learning rates
- Check if your data is correctly formatted

### Issue: "No module named ..."
**Solution**: Make sure all requirements are installed:
```bash
pip install -r requirements.txt
```

## Next Steps

1. **Tune hyperparameters**: Adjust learning rates, λ, γ, margin in Stage B
2. **Try different backbones**: ResNet-34 is faster, ResNet-50 is more accurate
3. **Implement doctor feedback**: Use the `doctor_interaction.py` module
4. **Visualize results**: Use visualization functions to understand model behavior

## Need Help?

- Check the full [README.md](README.md) for detailed documentation
- Look at [demo.py](demo.py) for usage examples
- Review the paper equations in the README
- Open an issue on GitHub

## Minimal Working Example

If you just want to test the model without data:

```python
from model import CSRModel
import torch

# Create model
model = CSRModel(num_concepts=10, num_classes=3, num_prototypes_per_concept=5)

# Random input
x = torch.randn(2, 3, 224, 224)

# Stage A: Concept prediction
out_a = model(x, stage='concept')
print(f"Concept logits shape: {out_a['concept_logits'].shape}")  # (2, 10)

# Stage B: Get concept vectors
concept_labels = torch.randint(0, 2, (2, 10)).float()
out_b = model(x, stage='prototype', concept_labels=concept_labels)
print(f"Projected vectors shape: {out_b['v_proj'].shape}")  # (2, 10, 128)

# Stage C: Task classification
out_c = model(x, stage='task')
print(f"Task logits shape: {out_c['task_logits'].shape}")  # (2, 3)
print(f"Similarities shape: {out_c['similarities'].shape}")  # (2, 10, 5)

print("\n✓ Model works!")
```

Happy coding! 🎉
