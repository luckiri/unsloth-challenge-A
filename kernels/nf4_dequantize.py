from kernels.nf4_dequantize import test_dequantize, your_dequantize_nf4

unsloth_time = test_dequantize(unsloth_dequantize)  # 5.38s
our_time = test_dequantize(your_dequantize_nf4)     # 1.28s
print(f"Speedup: {unsloth_time/our_time:.1f}x")     # Speedup: 4.2x
