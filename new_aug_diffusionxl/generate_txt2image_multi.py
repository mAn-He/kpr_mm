import os
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import gc
import logging

# 모델의 프로그레스바를 비활성화하기 위한 설정
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# 프로세스당 하나의 GPU를 사용하는 독립적인 worker 함수
def worker_process(gpu_id, image_files, input_dir, mask_dir, output_dir):
    try:
        # GPU 설정
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        print(f"Worker {gpu_id} started on {device}")
        
        # 모델 초기화 (각 프로세스가 독립적으로 자신의 GPU에 모델 로드)
        print(f"Worker {gpu_id}: Initializing SDXL model...")
        
        # 모델 로드 시 출력 숨기기
        inpaint_model = StableDiffusionXLInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        inpaint_model.to(device)
        
        # diffusers의 진행 표시줄 비활성화
        inpaint_model.set_progress_bar_config(disable=True)
        
        # 메모리 최적화
        inpaint_model.enable_attention_slicing()
        inpaint_model.enable_vae_slicing()
        
        # 결과 디렉토리
        os.makedirs(output_dir, exist_ok=True)
        
        # 캠퍼스 환경 프롬프트
        prompts = [
            "University campus pathway, brick buildings background, sunny daylight, distant pedestrians.",
            "Paved walkway through green campus lawn, lined with trees, clear blue sky.",
            "Modern university plaza, concrete ground, glass building facade distance, students walking.",
            "Campus sidewalk beside building with large windows, trees bordering, bright daytime.",
            "Outdoor university campus scene, mix of architecture and nature, bright daylight."
        ]
        
        # 이미 처리된 파일 확인
        processed_files = []
        new_files_to_process = []
        
        for image_file in image_files:
            original_filename = os.path.splitext(os.path.basename(image_file))[0]
            
            # 프롬프트 중 하나만 확인해도 모두 처리된 것으로 간주 (일관성 위해)
            check_file = os.path.join(output_dir, f"{original_filename}_augmented_1.jpg")
            
            if os.path.exists(check_file):
                processed_files.append(image_file)
            else:
                new_files_to_process.append(image_file)
        
        # 이미 처리된 파일 수 출력
        print(f"Worker {gpu_id}: {len(processed_files)} files already processed, {len(new_files_to_process)} files to process")
        
        # 처리할 파일이 없으면 종료
        if len(new_files_to_process) == 0:
            print(f"Worker {gpu_id}: No new files to process")
            return
        
        # tqdm으로 전체 진행 상황 표시 (처리할 파일만)
        print(f"Worker {gpu_id}: Processing {len(new_files_to_process)} images...")
        pbar = tqdm(total=len(new_files_to_process), desc=f"Worker {gpu_id}", position=gpu_id)
        
        for image_file in new_files_to_process:
            try:
                img_path = os.path.join(input_dir, image_file)
                
                # 이미지 로드
                img = cv2.imread(img_path)
                if img is None:
                    pbar.update(1)
                    continue
                
                original_filename = os.path.splitext(os.path.basename(img_path))[0]
                
                # 마스크 선택 및 검증
                mask_path = os.path.join(mask_dir, f"{original_filename}_mask.png")
                if not os.path.exists(mask_path):
                    pbar.update(1)
                    continue
                
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None or np.all(mask == 0):
                    pbar.update(1)
                    continue
                
                inverted_mask = cv2.bitwise_not(mask)
                
                # 각 프롬프트에 대한 이미지 생성
                for idx, prompt in enumerate(prompts):
                    try:
                        # 해당 프롬프트로 이미 처리된 파일 확인
                        save_path = os.path.join(output_dir, f"{original_filename}_augmented_{idx+1}.jpg")
                        if os.path.exists(save_path):
                            continue  # 이미 처리된 프롬프트는 건너뛰기
                            
                        # RGB로 변환
                        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        mask_pil = Image.fromarray(inverted_mask)
                        
                        # 인페인팅 수행 (출력 감춤)
                        result = inpaint_model(
                            prompt=prompt,
                            image=img_pil,
                            mask_image=mask_pil,
                            num_inference_steps=20,
                            guidance_scale=7.5,
                            strength=0.7,
                        ).images[0]
                        
                        # BGR로 변환 및 리사이즈
                        result_cv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
                        result_resized = cv2.resize(
                            result_cv,
                            (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_LINEAR
                        )
                        
                        # 결과가 유효한지 확인
                        if result_resized is not None and not np.all(result_resized == 0):
                            # 이미지 저장
                            cv2.imwrite(save_path, result_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                            
                    except Exception as e:
                        # 에러 로그도 최소화
                        pass
                
                # 메모리 정리
                gc.collect()
                torch.cuda.empty_cache()
                
                # 진행 표시줄 업데이트
                pbar.update(1)
                
            except Exception as e:
                pbar.update(1)
        
        # 진행 표시줄 닫기
        pbar.close()
        print(f"Worker {gpu_id}: Finished processing all assigned images")
        
    except Exception as e:
        print(f"Worker {gpu_id}: Fatal error: {e}")
        raise e

def main():
    # 사용 가능한 GPU 수 확인
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs available. Exiting.")
        return
    
    print(f"Found {num_gpus} GPUs")
    
    # 경로 설정
    input_dir = "/scratch/ghtmd9277/keypoint_promptable_reidentification/Market-1501-v15.09.15/bounding_box_train"
    mask_dir = "/scratch/ghtmd9277/keypoint_promptable_reidentification/Market-1501-v15.09.15/market_mask"
    output_dir = "/scratch/ghtmd9277/keypoint_promptable_reidentification/Market-1501-v15.09.15/aug_market_inpainting_5prompts"
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 파일 가져오기
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    # 결정적 분배를 위한 정렬
    image_files.sort()
    
    print(f"Total images to process: {len(image_files)}")
    
    # GPU별로 작업 분배
    gpu_batches = []
    batch_size = len(image_files) // num_gpus
    
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * batch_size
        # 마지막 GPU가 남은 모든 이미지를 처리하도록 설정
        end_idx = (gpu_id + 1) * batch_size if gpu_id < num_gpus - 1 else len(image_files)
        gpu_batches.append(image_files[start_idx:end_idx])
        print(f"GPU {gpu_id}: Assigned {len(gpu_batches[-1])} images (index {start_idx} to {end_idx-1})")
    
    # 멀티프로세싱 시작 방법 설정
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 이미 설정되어 있으면 무시
    
    # 프로세스 생성 및 시작
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, gpu_batches[gpu_id], input_dir, mask_dir, output_dir)
        )
        p.start()
        processes.append(p)
    
    # 모든 프로세스 완료 대기
    for p in processes:
        p.join()
    
    print("All processes completed. Augmentation complete.")

if __name__ == "__main__":
    # 파이썬 로깅 레벨을 ERROR로 설정하여 불필요한 출력 최소화
    logging.basicConfig(level=logging.ERROR)
    main()