import torch
import numpy as np

def inspect():
    dump_path = "/home/patemotter_google_com/workspace/vllm-torchtpu/scratch/nan_dump_model.layers.0.attn_rank0.pt"
    print(f"Loading {dump_path} ...")
    data = torch.load(dump_path, map_location="cpu")
    
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            has_nan = torch.isnan(v).any().item() if v.dtype.is_floating_point else False
            has_inf = torch.isinf(v).any().item() if v.dtype.is_floating_point else False
            print(f"Key: {k:<25} | Shape: {str(list(v.shape)):<25} | Dtype: {v.dtype} | Has NaN: {has_nan:<5} | Has Inf: {has_inf:<5}")
            if k == "swa_seq_lens":
                print("    Values: ", v.tolist())
            elif k == "swa_block_tables":
                print("    Values: ", v[:32].tolist())
            if k in ("q", "kv"):
                vf = v.to(torch.float32)
                print(f"    Stats: min={vf.min().item():.6f} | max={vf.max().item():.6f} | mean={vf.mean().item():.6f}")
            if has_nan or has_inf:
                # Cast to f32
                vf = v.to(torch.float32)
                nan_mask = torch.isnan(vf)
                inf_mask = torch.isinf(vf)
                print(f"    NaN count: {nan_mask.sum().item()} | Inf count: {inf_mask.sum().item()}")
                # If key is attn_sink, print all elements
                if k == "attn_sink":
                    print("    Values: ", vf.tolist())
                else:
                    print(f"    Non-NaN min: {vf[~nan_mask & ~inf_mask].min().item():.6f} | max: {vf[~nan_mask & ~inf_mask].max().item():.6f} | mean: {vf[~nan_mask & ~inf_mask].mean().item():.6f}")
        else:
            print(f"Key: {k:<25} | Value: {v}")

if __name__ == "__main__":
    try:
        inspect()
    except Exception as e:
        print("inspect() failed with error:", e)
    
    # Run dequantization check
    import torch
    import numpy as np
    import ml_dtypes
    
    data = torch.load("/home/patemotter_google_com/workspace/vllm-torchtpu/scratch/nan_dump_model.layers.0.attn_rank0.pt", map_location="cpu")
    if "sw_cache" in data:
        sw_cache = data["sw_cache"].squeeze(2) # shape [2894, 16, 4, 640]
        # Check page 1
        page_idx = 1
        page_data = sw_cache[page_idx]
        bkv = page_data.reshape(64, 640)
        bkv_np = bkv.numpy()
        
        # Search for non-zero pages in sw_cache
        non_zero_pages = []
        for p in range(sw_cache.shape[0]):
            if sw_cache[p].any():
                non_zero_pages.append(p)
        print(f"Non-zero pages in sw_cache: {non_zero_pages}")
        
        # If there are non-zero pages, print statistics for the first one
        if non_zero_pages:
            first_page = non_zero_pages[0]
            first_page_data = sw_cache[first_page].reshape(64, 640).numpy()
            first_page_nope = first_page_data[:, :448]
            print(f"First non-zero page {first_page} nope bytes: min={first_page_nope.min()} | max={first_page_nope.max()} | mean={first_page_nope.mean():.3f}")
            
            # Check if any byte is 0x7F or 0xFF (NaN in float8_e4m3fn)
            nan_bytes_mask = (first_page_nope == 0x7F) | (first_page_nope == 0xFF)
            print(f"First non-zero page {first_page} nope bytes: Has 0x7F/0xFF: {nan_bytes_mask.any()} | Count: {nan_bytes_mask.sum()}")
            
            # Bitcast to float8_e4m3fn
            nope_fp8 = first_page_nope.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
            print(f"First non-zero page {first_page} nope view: Has NaN: {np.isnan(nope_fp8).any()} | Count: {np.isnan(nope_fp8).sum()}")
            
            # Let's check first 15 tokens (indices 0 to 14) which are valid for sequence 0
            print(f"Valid 15 tokens (0-14) nope_fp8 view: Has NaN: {np.isnan(nope_fp8[:15]).any()} | Count: {np.isnan(nope_fp8[:15]).sum()}")
            
            first_page_scales = first_page_data[:, 576:583]
            print(f"First non-zero page {first_page} scales: min={first_page_scales.min()} | max={first_page_scales.max()} | mean={first_page_scales.mean():.3f}")
            
            # Decode scales
            scales_decoded = np.power(2.0, first_page_scales.astype(np.float32) - 127.0)
            print(f"First non-zero page {first_page} decoded scales: min={scales_decoded.min()} | max={scales_decoded.max()} | mean={scales_decoded.mean():.6e}")
            
            # Check if decoded scales contain NaN or Inf
            print(f"Decoded scales Has NaN: {np.isnan(scales_decoded).any()} | Has Inf: {np.isinf(scales_decoded).any()}")
            
            # Extract and reconstruct rope
            rope_bytes = first_page_data[:, 448:576]
            rope_bytes_reshaped = rope_bytes.reshape(64, 64, 2)
            byte0 = rope_bytes_reshaped[:, :, 0].astype(np.uint32)
            byte1 = rope_bytes_reshaped[:, :, 1].astype(np.uint32)
            rope_uint16 = byte0 | (byte1 << 8)
            rope_float32_bits = rope_uint16 << 16
            
            # Reconstruct float32 by interpreting bits as float32
            # We can use np.frombuffer on tobytes() to convert the uint32 array to float32
            rope_f32 = np.frombuffer(rope_float32_bits.tobytes(), dtype=np.float32).reshape(64, 64)
            print(f"Reconstructed rope: min={rope_f32.min():.6f} | max={rope_f32.max():.6f} | mean={rope_f32.mean():.6f}")
            print(f"Reconstructed rope Has NaN: {np.isnan(rope_f32).any()} | Has Inf: {np.isinf(rope_f32).any()}")
            # Print NaN count if any
            if np.isnan(rope_f32).any():
                print(f"Reconstructed rope NaN count: {np.isnan(rope_f32).sum()}")
