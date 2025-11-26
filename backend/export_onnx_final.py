import torch
import os
import sys
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet

def export_real_esrgan_to_onnx():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "/app/esrgan/RealESRGAN_x4plus.pth"
    OUTPUT_ONNX_PATH = "RealESRGAN_x4plus.onnx"
    
    # paths.py 파일에 의존하지 않도록 수동으로 경로를 추가했던 로직은 삭제합니다. 
    # basicsr이 정상적으로 설치되었다면 바로 import 되어야 합니다.
    
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model.load_state_dict(torch.load(MODEL_PATH)['params'], strict=True)
    model.eval()
    model.to(DEVICE)
    
    dummy_input = torch.randn(1, 3, 64, 64, device=DEVICE)
    print(f"모델 아키텍처 로드 완료. ONNX로 내보내기 시작...")
    
    torch.onnx.export(
        model, 
        dummy_input, 
        OUTPUT_ONNX_PATH, 
        export_params=True, 
        opset_version=17, 
        do_constant_folding=True,
        input_names=['input_image'], 
        output_names=['output_image'],
        dynamic_axes={'input_image': {0: 'batch_size'},
                      'output_image': {0: 'batch_size'}}
    )
    print(f"✅ ONNX 모델 내보내기 성공: {OUTPUT_ONNX_PATH}")
    print("-------------------------------------------------------")

if __name__ == "__main__":
    export_real_esrgan_to_onnx()
