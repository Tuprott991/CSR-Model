"""
Demo script showing how to use CSR model for inference and interaction
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from model import CSRModel
from doctor_interaction import DoctorInteraction, convert_image_boxes_to_feature_boxes


def load_and_preprocess_image(image_path: str, input_size: int = 224) -> tuple:
    """
    Load and preprocess image for CSR model
    
    Returns:
        preprocessed_tensor: (1, 3, H, W) for model input
        original_image: (H, W, 3) numpy array for visualization
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    
    # Transform for model
    transform = transforms.Compose([
        transforms.Resize(int(input_size * 1.1)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    preprocessed = transform(image).unsqueeze(0)  # Add batch dimension
    
    return preprocessed, original_image


def demo_basic_inference():
    """
    Demo 1: Basic inference with CSR model
    """
    print("=" * 60)
    print("DEMO 1: Basic Inference")
    print("=" * 60)
    
    # Configuration (adjust based on your dataset)
    num_concepts = 15  # Number of medical findings
    num_classes = 3    # Number of diseases
    num_prototypes = 10
    
    # Create model
    model = CSRModel(
        num_concepts=num_concepts,
        num_classes=num_classes,
        num_prototypes_per_concept=num_prototypes,
        backbone='resnet50',
        proj_dim=128,
        pretrained=True
    )
    
    # Load trained weights (replace with your checkpoint path)
    # checkpoint = torch.load('checkpoints/stage_c_best.pth', map_location='cpu')
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Load and preprocess image
    # image_tensor, original_image = load_and_preprocess_image('path/to/your/image.jpg')
    # For demo, use random image
    image_tensor = torch.randn(1, 3, 224, 224).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(image_tensor, stage='task')
        task_logits = outputs['task_logits']
        similarities = outputs['similarities']
        
        # Get predictions
        probs = F.softmax(task_logits, dim=1)
        pred_class = probs.argmax(dim=1).item()
        pred_confidence = probs[0, pred_class].item()
        
        print(f"\nPredicted Class: {pred_class}")
        print(f"Confidence: {pred_confidence:.4f}")
        print(f"\nAll Class Probabilities:")
        for i, prob in enumerate(probs[0]):
            print(f"  Class {i}: {prob.item():.4f}")
        
        # Get top activated concepts
        concept_scores = similarities.max(dim=2)[0].squeeze(0)
        top_k = 5
        top_values, top_indices = torch.topk(concept_scores, top_k)
        
        print(f"\nTop {top_k} Activated Concepts:")
        for i, (idx, score) in enumerate(zip(top_indices, top_values)):
            print(f"  {i+1}. Concept {idx.item()}: {score.item():.4f}")


def demo_concept_rejection():
    """
    Demo 2: Doctor-in-the-loop concept rejection
    """
    print("\n" + "=" * 60)
    print("DEMO 2: Concept Rejection")
    print("=" * 60)
    
    # Create model
    model = CSRModel(
        num_concepts=15,
        num_classes=3,
        num_prototypes_per_concept=10
    )
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Create interaction interface
    doctor = DoctorInteraction(model, device=device)
    
    # Load image
    image_tensor = torch.randn(1, 3, 224, 224).to(device)
    
    # Initial prediction
    print("\n--- Initial Prediction ---")
    with torch.no_grad():
        outputs = model(image_tensor, stage='task')
        initial_logits = outputs['task_logits']
        initial_probs = F.softmax(initial_logits, dim=1)
        print(f"Predicted Class: {initial_probs.argmax(dim=1).item()}")
        print(f"Confidence: {initial_probs.max().item():.4f}")
    
    # Get top concepts
    top_concepts = doctor.get_top_concepts(image_tensor, top_k=5)
    print("\nTop Activated Concepts:")
    for concept_id, score in top_concepts.items():
        print(f"  Concept {concept_id}: {score:.4f}")
    
    # Reject some concepts (e.g., doctor thinks they're incorrect)
    rejected_concepts = [2, 5]  # Example: reject concepts 2 and 5
    print(f"\n--- After Rejecting Concepts {rejected_concepts} ---")
    
    modified_logits = doctor.reject_concepts(image_tensor, rejected_concepts)
    modified_probs = F.softmax(modified_logits, dim=1)
    
    print(f"Predicted Class: {modified_probs.argmax(dim=1).item()}")
    print(f"Confidence: {modified_probs.max().item():.4f}")
    
    print("\nPrediction changed!" if initial_probs.argmax() != modified_probs.argmax() 
          else "\nPrediction stayed the same")


def demo_spatial_feedback():
    """
    Demo 3: Spatial-level feedback with bounding boxes
    """
    print("\n" + "=" * 60)
    print("DEMO 3: Spatial Feedback")
    print("=" * 60)
    
    # Create model
    model = CSRModel(
        num_concepts=15,
        num_classes=3,
        num_prototypes_per_concept=10
    )
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Create interaction interface
    doctor = DoctorInteraction(model, device=device)
    
    # Load image
    image_tensor = torch.randn(1, 3, 224, 224).to(device)
    
    # Initial prediction
    print("\n--- Initial Prediction ---")
    with torch.no_grad():
        outputs = model(image_tensor, stage='task')
        initial_logits = outputs['task_logits']
        initial_probs = F.softmax(initial_logits, dim=1)
        print(f"Predicted Class: {initial_probs.argmax(dim=1).item()}")
        print(f"Confidence: {initial_probs.max().item():.4f}")
    
    # Get feature map size
    f = model.feature_extractor(image_tensor)
    _, _, H_feat, W_feat = f.shape
    print(f"\nFeature map size: {H_feat}x{W_feat}")
    
    # Define bounding boxes in image coordinates
    image_size = (224, 224)
    positive_boxes_img = [(50, 50, 150, 150)]  # Region of interest
    negative_boxes_img = [(10, 10, 40, 40)]    # Region to suppress
    
    # Convert to feature map coordinates
    positive_boxes = convert_image_boxes_to_feature_boxes(
        positive_boxes_img, image_size, (H_feat, W_feat)
    )
    negative_boxes = convert_image_boxes_to_feature_boxes(
        negative_boxes_img, image_size, (H_feat, W_feat)
    )
    
    print(f"\nPositive boxes (feature coords): {positive_boxes}")
    print(f"Negative boxes (feature coords): {negative_boxes}")
    
    # Inference with spatial feedback
    print("\n--- After Spatial Feedback (α=0.2) ---")
    modified_logits = doctor.inference_with_spatial_feedback(
        image_tensor,
        positive_boxes=positive_boxes,
        negative_boxes=negative_boxes,
        alpha=0.2
    )
    
    modified_probs = F.softmax(modified_logits, dim=1)
    print(f"Predicted Class: {modified_probs.argmax(dim=1).item()}")
    print(f"Confidence: {modified_probs.max().item():.4f}")
    
    # Compare probabilities
    print("\nProbability Changes:")
    for i in range(model.num_classes):
        initial = initial_probs[0, i].item()
        modified = modified_probs[0, i].item()
        change = modified - initial
        print(f"  Class {i}: {initial:.4f} → {modified:.4f} ({change:+.4f})")


def demo_interactive_diagnosis():
    """
    Demo 4: Complete interactive diagnosis workflow
    """
    print("\n" + "=" * 60)
    print("DEMO 4: Interactive Diagnosis")
    print("=" * 60)
    
    # Create model
    model = CSRModel(
        num_concepts=15,
        num_classes=3,
        num_prototypes_per_concept=10
    )
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Create interaction interface
    doctor = DoctorInteraction(model, device=device)
    
    # Define concept and class names
    concept_names = [
        'infiltration', 'effusion', 'consolidation', 'pneumothorax',
        'edema', 'emphysema', 'fibrosis', 'thickening', 'calcification',
        'cardiomegaly', 'nodule', 'mass', 'atelectasis', 'cavity', 'fracture'
    ]
    class_names = ['tuberculosis', 'pneumonia', 'normal']
    
    # Load image
    image_tensor = torch.randn(1, 3, 224, 224).to(device)
    
    # Get interactive diagnosis
    result = doctor.interactive_diagnosis(
        image_tensor,
        concept_names=concept_names,
        class_names=class_names
    )
    
    # Display results
    print("\n--- Diagnosis Results ---")
    print(f"Predicted Diagnosis: {class_names[result['predicted_class']]}")
    
    print("\nAll Diagnoses (sorted by probability):")
    for pred in result['predictions'][:3]:
        print(f"  {pred['name']}: {pred['probability']:.4f}")
    
    print("\nTop Contributing Concepts:")
    for i, concept in enumerate(result['top_concepts'][:5]):
        print(f"  {i+1}. {concept['name']}: {concept['score']:.4f}")
    
    print("\n--- Interpretation ---")
    print("The model's prediction is based on the presence of the above concepts.")
    print("A doctor can:")
    print("  1. Reject incorrect concepts to refine the diagnosis")
    print("  2. Draw boxes to indicate important/unimportant regions")
    print("  3. Visualize which image regions activate each concept")


def demo_visualization():
    """
    Demo 5: Visualize concept activations and similarity maps
    """
    print("\n" + "=" * 60)
    print("DEMO 5: Visualization")
    print("=" * 60)
    
    # Create model
    model = CSRModel(
        num_concepts=15,
        num_classes=3,
        num_prototypes_per_concept=10
    )
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Create interaction interface
    doctor = DoctorInteraction(model, device=device)
    
    # Load image
    image_tensor = torch.randn(1, 3, 224, 224).to(device)
    
    print("\nGenerating visualizations...")
    print("(In a real scenario, these would show heatmaps overlaid on the image)")
    
    # Visualize concept activation
    concept_id = 3
    print(f"\n1. Concept Activation Map for Concept {concept_id}")
    cam_viz = doctor.visualize_concept_activation(
        image_tensor,
        concept_id=concept_id
    )
    print(f"   Shape: {cam_viz.shape}")
    print("   This shows where in the image the concept is most activated")
    
    # Visualize similarity map
    prototype_id = 0
    print(f"\n2. Similarity Map for Concept {concept_id}, Prototype {prototype_id}")
    sim_viz = doctor.visualize_similarity_maps(
        image_tensor,
        concept_id=concept_id,
        prototype_id=prototype_id
    )
    print(f"   Shape: {sim_viz.shape}")
    print("   This shows which image regions are most similar to the prototype")
    
    print("\n✓ Visualizations generated successfully!")
    print("  (Use plt.imshow() to display them in a real application)")


def main():
    """
    Run all demos
    """
    print("\n" + "=" * 60)
    print("CSR MODEL DEMOS")
    print("=" * 60)
    print("\nThese demos show how to use the CSR model for:")
    print("  1. Basic inference")
    print("  2. Concept rejection (doctor-in-the-loop)")
    print("  3. Spatial feedback with bounding boxes")
    print("  4. Interactive diagnosis")
    print("  5. Visualization of activations")
    
    try:
        demo_basic_inference()
        demo_concept_rejection()
        demo_spatial_feedback()
        demo_interactive_diagnosis()
        demo_visualization()
        
        print("\n" + "=" * 60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nTo use with real data:")
        print("  1. Train the model using train.py")
        print("  2. Load your checkpoint in these demos")
        print("  3. Use load_and_preprocess_image() with your images")
        print("  4. Provide appropriate concept_names and class_names")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        print("Note: These demos use random tensors for illustration.")
        print("For real usage, train the model and load a checkpoint.")


if __name__ == '__main__':
    main()
