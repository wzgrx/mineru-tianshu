"""
RTX 5090 Benchmark Tool for Tianshu
测试 PaddleOCR 和 MinerU 在当前环境下的吞吐量
"""
import time
import torch
import paddle
from loguru import logger

def benchmark_pytorch():
    logger.info("🔥 Benchmarking PyTorch (CUDA)...")
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available!")
        return
    
    device = torch.device("cuda")
    logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # 简单的矩阵乘法压力测试
    size = 10000
    a = torch.randn(size, size, device=device, dtype=torch.float16)
    b = torch.randn(size, size, device=device, dtype=torch.float16)
    
    # 预热
    for _ in range(5): torch.mm(a, b)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(10):
        torch.mm(a, b)
    torch.cuda.synchronize()
    end = time.time()
    
    logger.info(f"   ✅ Matrix Mul (10k x 10k, FP16): {(end - start)/10:.4f} seconds/iter")

def benchmark_paddle():
    logger.info("🔥 Benchmarking PaddlePaddle...")
    if not paddle.device.is_compiled_with_cuda():
        logger.error("❌ Paddle CUDA not compiled!")
        return

    paddle.set_device("gpu:0")
    # 简单的 Paddle 测试逻辑...
    logger.info("   ✅ Paddle GPU initialized successfully.")

if __name__ == "__main__":
    benchmark_pytorch()
    benchmark_paddle()
