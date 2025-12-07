"""
HRIBench 테스트 스크립트 - 추론과 시각화를 함께 실행합니다.

개별 실행:
    - 추론만 실행: python run_inference.py
    - GIF 생성만 실행: python generate_trajectory_gif.py

이 스크립트는 두 작업을 순차적으로 실행합니다:
    python test.py
"""
from run_inference import run_inference
from generate_trajectory_gif import generate_gifs


if __name__ == "__main__":
    # Step 1: Run inference on dataset
    print("=" * 60)
    print("Step 1: Running inference on HRIBench dataset")
    print("=" * 60)
    run_inference()
    
    # Step 2: Generate trajectory GIFs
    print("\n" + "=" * 60)
    print("Step 2: Generating trajectory GIFs")
    print("=" * 60)
    generate_gifs()
    
    print("\n" + "=" * 60)
    print("All tasks completed!")
    print("=" * 60)
