"""
HRIBench 데이터셋에 대해 모델 추론을 실행하고 결과를 저장하는 스크립트.

Usage:
    python run_inference.py
    python run_inference.py --output custom_name.json
    python run_inference.py --dataset /path/to/dataset
    python run_inference.py --dataset /path/to/dataset --model pi05_droid

출력 파일명 형식: {모델이름}_{데이터셋이름}_{현재시간}.json
예: pi05_droid_hroi_20251207_143052.json
"""
import json
from datetime import datetime
from pathlib import Path

from PIL import Image

from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config

import numpy as np

# Available models
AVAILABLE_MODELS = {
    "pi0_base": "gs://openpi-assets/checkpoints/pi0_base",
    "pi0_fast_base": "gs://openpi-assets/checkpoints/pi0_fast_base",
    "pi05_base": "gs://openpi-assets/checkpoints/pi05_base",
    "pi05_droid": "gs://openpi-assets/checkpoints/pi05_droid",
}

# Dataset path
SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR / "dataset" / "hribench" / "hroi"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_image(image_path: Path) -> np.ndarray:
    """Load and preprocess image to 224x224x3 uint8 format."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    return np.array(img, dtype=np.uint8)


def load_prompt(prompt_path: Path) -> str:
    """Load prompt from droid.txt file."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def get_demo_dirs(dataset_dir: Path) -> list[Path]:
    """Get all demo directories sorted by number."""
    demo_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("demo_")]
    # Sort by demo number
    demo_dirs.sort(key=lambda x: int(x.name.split("_")[1]))
    return demo_dirs


def generate_output_filename(model_name: str, dataset_path: Path) -> str:
    """
    Generate output filename in format: {model_name}_{dataset_name}_{timestamp}.json
    
    Args:
        model_name: Name of the model (e.g., "pi05_droid")
        dataset_path: Path to the dataset directory
    
    Returns:
        Generated filename string
    """
    dataset_name = dataset_path.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{dataset_name}_{timestamp}.json"


def run_inference(output_file: str = None, dataset_dir: str = None, model_name: str = "pi05_droid", repeat: int = 1):
    """
    Run inference on all demos in hribench dataset.
    
    Args:
        output_file: Name of the output JSON file (if None, auto-generated)
        dataset_dir: Path to the dataset directory (default: dataset/hribench/hroi)
        model_name: Name of the model to use (default: pi05_droid)
    """
    # Determine dataset directory
    if dataset_dir is not None:
        data_path = Path(dataset_dir)
    else:
        data_path = DATASET_DIR
    
    if not data_path.exists():
        print(f"Error: Dataset directory not found at {data_path}")
        return None
    
    # Validate model name
    if model_name not in AVAILABLE_MODELS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {list(AVAILABLE_MODELS.keys())}")
        return None
    
    
    # Configuration
    config = _config.get_config(model_name)
    
    checkpoint_url = AVAILABLE_MODELS[model_name]
    checkpoint_dir = download.maybe_download(checkpoint_url)
    
    # Create a trained policy
    print(f"Loading model: {model_name}...")
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    print("Model loaded successfully!")
    
    demo_dirs = get_demo_dirs(data_path)
    print(f"Found {len(demo_dirs)} demos in {data_path}")
    
    output_paths = []

    for repeat_idx in range(repeat):
        if repeat > 1:
            print(f"\n{'#'*50}")
            print(f"# Repeat {repeat_idx + 1}/{repeat}")
            print(f"{'#'*50}")
        
        # Generate output filename for each repeat (always generate new timestamp)
        if output_file is not None:
            # User specified filename - add repeat index if repeating
            if repeat > 1:
                base_name = Path(output_file).stem
                current_output_file = f"{base_name}_run{repeat_idx + 1}.json"
            else:
                current_output_file = output_file
        else:
            # Auto-generate filename with current timestamp
            current_output_file = generate_output_filename(model_name, data_path)
        
        print(f"Output file: {current_output_file}")
        
        results = {}
        
        for idx, demo_dir in enumerate(demo_dirs):
            demo_name = demo_dir.name
            image_path = demo_dir / "wrong_answers" / "processed_frame.png"
            prompt_path = demo_dir / "droid.txt"
            
            # Check if required files exist
            if not image_path.exists():
                print(f"Skipping {demo_name}: image not found at {image_path}")
                continue
            if not prompt_path.exists():
                print(f"Skipping {demo_name}: prompt not found at {prompt_path}")
                continue
            
            # Load image and prompt
            image = load_image(image_path)
            prompt = load_prompt(prompt_path)
            
            print(f"[{idx+1}/{len(demo_dirs)}] Processing {demo_name}...")
            print(f"  Prompt: {prompt[:80]}...")
            
            # Fixed initial values at origin (0, 0, 0, ...)
            initial_joint_position = [0.0] * 7
            initial_gripper_position = [0.0]
            
            # Create example for inference
            example = {
                "observation/exterior_image_1_left": image,
                "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),  # dummy wrist image
                "observation/joint_position": np.array(initial_joint_position),
                "observation/gripper_position": np.array(initial_gripper_position),
                "prompt": prompt
            }
            
            # Run inference
            result = policy.infer(example)
            
            # Store result with initial values
            results[demo_name] = {
                "prompt": prompt,
                "initial_state": {
                    "joint_position": initial_joint_position,
                    "gripper_position": initial_gripper_position
                },
                "actions_shape": list(result["actions"].shape),
                "actions": result["actions"].tolist()
            }
            
            print(f"  Actions shape: {result['actions'].shape}")
        
        # Save results
        output_path = OUTPUT_DIR / current_output_file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        output_paths.append(output_path)
        
        print(f"\n{'='*50}")
        print(f"Results saved to {output_path}")
        print(f"Total processed: {len(results)} demos")
        print(f"{'='*50}")
    
    # Delete the policy to free up memory
    del policy
    
    if repeat > 1:
        print(f"\n{'#'*50}")
        print(f"# All {repeat} runs completed!")
        print(f"# Output files:")
        for p in output_paths:
            print(f"#   - {p}")
        print(f"{'#'*50}")
    
    return output_paths if repeat > 1 else output_paths[0]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run inference on HRIBench dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_inference.py
    python run_inference.py --dataset /path/to/dataset
    python run_inference.py --model pi05_droid --dataset ./my_dataset
    python run_inference.py --output custom_name.json

Output filename format (auto-generated):
    {model_name}_{dataset_name}_{timestamp}.json
    Example: pi05_droid_hroi_20251207_143052.json
        """
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON filename (default: auto-generated as {model}_{dataset}_{timestamp}.json)"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help="Path to dataset directory containing demo_* folders (default: dataset/hribench/hroi)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="pi05_droid",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Model to use for inference (default: pi05_droid)"
    )
    parser.add_argument(
        "--repeat", "-r",
        type=int,
        default=1,
        help="Number of times to repeat the inference (default: 1)"
    )    
    args = parser.parse_args()
    run_inference(output_file=args.output, dataset_dir=args.dataset, model_name=args.model, repeat=args.repeat)

