UNQUANTIZED = "unquantized"
MXFP4 = "mxfp4"
AWQ = "awq"
FP8 = "fp8"
COMPRESSED_TENSORS = "compressed-tensors"
DEEPSEEK_V4_FP8 = "deepseek_v4_fp8"


def get_tpu_quant_method(quant_method: str) -> str:
    return "tpu-" + quant_method
