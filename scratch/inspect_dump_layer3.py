import torch
import numpy as np
import ml_dtypes


def inspect():
    dump_path = "/home/patemotter_google_com/workspace/vllm-torchtpu/scratch/nan_dump_model.layers.3.attn_rank0.pt"
    print(f"Loading {dump_path} ...")
    data = torch.load(dump_path, map_location="cpu")

    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            has_nan = torch.isnan(v).any().item() if v.dtype.is_floating_point else False
            has_inf = torch.isinf(v).any().item() if v.dtype.is_floating_point else False
            print(f"Key: {k:<25} | Shape: {str(list(v.shape)):<25} | Dtype: {v.dtype} | Has NaN: {has_nan!s:<5} | Has Inf: {has_inf!s:<5}")
            if k in ("positions", "compress_ratio", "main_kv_lens", "main_cu_q_lens", "main_distribution"):
                print("    Values: ", v.flatten().tolist())
            if k == "main_page_indices":
                print("    Values (first 32): ", v.flatten()[:32].tolist())
            if k == "extra":
                print("    Shape detail, first row: ", v[0].tolist() if v.ndim > 1 else v[:16].tolist())
        else:
            print(f"Key: {k:<25} | Value: {v}")

    # Locate exactly which (token, head, col) positions in output are NaN
    if "output" in data:
        out = data["output"].to(torch.float32)
        nan_mask = torch.isnan(out)
        print(f"\noutput NaN total: {nan_mask.sum().item()} / {out.numel()}, shape={list(out.shape)}")
        idxs = torch.nonzero(nan_mask)
        if idxs.shape[0] > 0:
            # Summarize unique columns (last dim) where NaN occurs
            last_dim_idx = idxs[:, -1]
            uniq_cols, counts = torch.unique(last_dim_idx, return_counts=True)
            print(f"NaN unique last-dim columns: {uniq_cols.tolist()} counts: {counts.tolist()}")
            print(f"First 10 NaN index tuples: {idxs[:10].tolist()}")

    positions = data.get("positions")
    main_page_indices = data.get("main_page_indices")
    main_cache_kv = data.get("main_cache_kv")
    compress_ratio = data.get("compress_ratio")
    print(f"\npositions={positions.flatten().tolist() if positions is not None else None}")
    print(f"compress_ratio={compress_ratio}")

    if main_cache_kv is not None:
        cache = main_cache_kv
        if cache.ndim == 5:
            cache = cache.squeeze(2)
        print(f"\nmain_cache_kv shape (post-squeeze): {list(cache.shape)}")

        non_zero_pages = []
        for p in range(cache.shape[0]):
            if cache[p].any():
                non_zero_pages.append(p)
        print(f"Non-zero pages in main_cache_kv: {non_zero_pages[:32]} (total {len(non_zero_pages)})")

        # Inspect page(s) referenced by main_page_indices for request 0
        if main_page_indices is not None:
            page_ids = main_page_indices.flatten().tolist()
            print(f"main_page_indices (req0, up to 16): {page_ids[:16]}")
            for pid in sorted(set(page_ids[:8])):
                if pid < 0 or pid >= cache.shape[0]:
                    continue
                page_data = cache[pid]
                bkv = page_data.reshape(-1, page_data.shape[-1]).numpy()
                nope = bkv[:, :448]
                nan_bytes_mask = (nope == 0x7F) | (nope == 0xFF)
                nope_fp8 = nope.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
                print(f"  page {pid}: nope raw 0x7F/0xFF byte count={nan_bytes_mask.sum()}, "
                      f"fp8-decoded NaN count={np.isnan(nope_fp8).sum()}, "
                      f"rows with any NaN={np.isnan(nope_fp8).any(axis=1).sum()}/{nope_fp8.shape[0]}")


if __name__ == "__main__":
    try:
        inspect()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("inspect() failed with error:", e)
