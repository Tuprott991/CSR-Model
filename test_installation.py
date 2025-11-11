"""
Test script to verify CSR model installation and components
Run this to ensure everything is working correctly
"""

import torch
import sys


def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    try:
        from model import CSRModel, FeatureExtractor, ConceptHead, Projector, PrototypeLayer, TaskHead
        from dataloader import MedicalImageDataset, TBX11KDataset, VinDrCXRDataset, ISICDataset
        from losses import MultiPrototypeContrastiveLoss, ConceptBCELoss, TaskCrossEntropyLoss
        from doctor_interaction import DoctorInteraction
        from utils import set_seed, count_parameters, AverageMeter
        print("✓ All imports successful!")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_model_creation():
    """Test if model can be created"""
    print("\nTesting model creation...")
    try:
        from model import CSRModel
        
        # Test ResNet-50
        print("  Testing ResNet-50 backbone...")
        model = CSRModel(
            num_concepts=10,
            num_classes=3,
            num_prototypes_per_concept=5,
            backbone='resnet50',
            proj_dim=128,
            pretrained=False  # Faster for testing
        )
        print(f"  ✓ ResNet-50 model created successfully!")
        print(f"    - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test ConvNeXtV2 if timm is available
        try:
            import timm
            print("  Testing ConvNeXtV2 backbone...")
            model_convnext = CSRModel(
                num_concepts=10,
                num_classes=3,
                num_prototypes_per_concept=5,
                backbone='convnextv2_base',
                proj_dim=128,
                pretrained=False
            )
            print(f"  ✓ ConvNeXtV2 model created successfully!")
            print(f"    - Total parameters: {sum(p.numel() for p in model_convnext.parameters()):,}")
        except ImportError:
            print("  ⚠ timm not installed, skipping ConvNeXtV2 test")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test forward pass through all stages"""
    print("\nTesting forward pass...")
    try:
        from model import CSRModel
        
        model = CSRModel(
            num_concepts=10,
            num_classes=3,
            num_prototypes_per_concept=5,
            pretrained=False
        )
        model.eval()
        
        # Create dummy input
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)
        
        # Test Stage A
        print("  Testing Stage A (Concept Head)...")
        with torch.no_grad():
            outputs_a = model(x, stage='concept')
            assert outputs_a['concept_logits'].shape == (batch_size, 10)
            assert outputs_a['cam'].shape[0] == batch_size
        print("  ✓ Stage A working")
        
        # Test Stage B
        print("  Testing Stage B (Prototypes)...")
        concept_labels = torch.randint(0, 2, (batch_size, 10)).float()
        with torch.no_grad():
            outputs_b = model(x, stage='prototype', concept_labels=concept_labels)
            assert outputs_b['v_proj'].shape == (batch_size, 10, 128)
        print("  ✓ Stage B working")
        
        # Test Stage C
        print("  Testing Stage C (Task Head)...")
        with torch.no_grad():
            outputs_c = model(x, stage='task')
            assert outputs_c['task_logits'].shape == (batch_size, 3)
            assert outputs_c['similarities'].shape == (batch_size, 10, 5)
        print("  ✓ Stage C working")
        
        print("✓ All forward passes successful!")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_losses():
    """Test loss functions"""
    print("\nTesting loss functions...")
    try:
        from losses import MultiPrototypeContrastiveLoss, ConceptBCELoss, TaskCrossEntropyLoss
        
        batch_size = 2
        num_concepts = 10
        num_classes = 3
        M = 5
        proj_dim = 128
        
        # Test Concept BCE Loss
        print("  Testing Concept BCE Loss...")
        concept_loss_fn = ConceptBCELoss()
        concept_logits = torch.randn(batch_size, num_concepts)
        concept_labels = torch.randint(0, 2, (batch_size, num_concepts)).float()
        loss = concept_loss_fn(concept_logits, concept_labels)
        assert loss.item() >= 0
        print("  ✓ Concept BCE Loss working")
        
        # Test Task CE Loss
        print("  Testing Task Cross-Entropy Loss...")
        task_loss_fn = TaskCrossEntropyLoss()
        task_logits = torch.randn(batch_size, num_classes)
        task_labels = torch.randint(0, num_classes, (batch_size,))
        loss = task_loss_fn(task_logits, task_labels)
        assert loss.item() >= 0
        print("  ✓ Task Cross-Entropy Loss working")
        
        # Test Multi-Prototype Contrastive Loss
        print("  Testing Multi-Prototype Contrastive Loss...")
        contrastive_loss_fn = MultiPrototypeContrastiveLoss()
        v_proj = torch.randn(batch_size, num_concepts, proj_dim)
        v_proj = torch.nn.functional.normalize(v_proj, p=2, dim=-1)
        prototypes = torch.randn(num_concepts, M, proj_dim)
        prototypes = torch.nn.functional.normalize(prototypes, p=2, dim=-1)
        concept_labels = torch.randint(0, 2, (batch_size, num_concepts)).float()
        # Ensure at least one positive label
        concept_labels[0, 0] = 1
        loss = contrastive_loss_fn(v_proj, concept_labels, prototypes)
        assert loss.item() >= 0
        print("  ✓ Multi-Prototype Contrastive Loss working")
        
        print("✓ All loss functions working!")
        return True
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_doctor_interaction():
    """Test doctor-in-the-loop functionality"""
    print("\nTesting doctor interaction...")
    try:
        from model import CSRModel
        from doctor_interaction import DoctorInteraction
        
        model = CSRModel(
            num_concepts=10,
            num_classes=3,
            num_prototypes_per_concept=5,
            pretrained=False
        )
        model.eval()
        
        doctor = DoctorInteraction(model, device='cpu')
        
        x = torch.randn(1, 3, 224, 224)
        
        # Test get top concepts
        print("  Testing get_top_concepts...")
        top_concepts = doctor.get_top_concepts(x, top_k=5)
        assert len(top_concepts) == 5
        print("  ✓ get_top_concepts working")
        
        # Test concept rejection
        print("  Testing concept rejection...")
        rejected_logits = doctor.reject_concepts(x, rejected_concept_ids=[2, 5])
        assert rejected_logits.shape == (1, 3)
        print("  ✓ concept rejection working")
        
        # Test interactive diagnosis
        print("  Testing interactive diagnosis...")
        result = doctor.interactive_diagnosis(x)
        assert 'predicted_class' in result
        assert 'top_concepts' in result
        assert 'predictions' in result
        print("  ✓ interactive diagnosis working")
        
        print("✓ Doctor interaction working!")
        return True
    except Exception as e:
        print(f"✗ Doctor interaction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cuda_availability():
    """Test CUDA availability"""
    print("\nTesting CUDA availability...")
    if torch.cuda.is_available():
        print(f"✓ CUDA is available!")
        print(f"  - Device count: {torch.cuda.device_count()}")
        print(f"  - Current device: {torch.cuda.current_device()}")
        print(f"  - Device name: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("⚠ CUDA is not available. Training will be slower on CPU.")
        return True  # Not a failure


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("CSR MODEL INSTALLATION TEST")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Loss Functions", test_losses),
        ("Doctor Interaction", test_doctor_interaction),
        ("CUDA", test_cuda_availability),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    
    print("=" * 60)
    print(f"Total: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Your CSR installation is working correctly.")
        print("\nNext steps:")
        print("  1. Prepare your dataset (see QUICKSTART.md)")
        print("  2. Run training: python train.py [args]")
        print("  3. Try demos: python demo.py")
        return 0
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        print("Make sure all requirements are installed: pip install -r requirements.txt")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
