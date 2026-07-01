"""Export Bonsai 4B model weights to GGUF format with ternary/binary quantization.

This script reads weights from the local models/ directory (same layout used by
download_model.sh) and writes .gguf files with custom bonsai-dit architecture
metadata.

Usage:
    python export_gguf.py --variant ternary --output bonsai-ternary.gguf
    python export_gguf.py --variant binary --output bonsai-binary.gguf
    python export_gguf.py --variant ternary --split  # Separate GGUF for DiT + VAE
"""

from __future__ import annotations

import argparse
import logging
import struct
import sys
from pathlib import Path

import numpy as np

# Try to import gguf-py
try:
    import gguf
    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False
    print("WARNING: gguf-py not installed. Install with: pip install gguf")
    print("Falling back to manual GGUF writing.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger("export_gguf")

DEMO_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = DEMO_DIR / "models"

# GGUF constants
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 8

# Tensor types
class GgufTensorType:
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q8_0 = 9
    I16 = 25
    I32 = 26
    # Custom ternary/binary types
    TQ1_0 = 34  # Ternary {-1, 0, 1}
    TQ2_0 = 35  # 2-bit ternary
    IQ1_S = 20  # Binary {-1, 1}
    IQ1_M = 29  # Binary {-1, 0, 1}

# Metadata types
class GgufMetadataType:
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12

# Bonsai architecture metadata keys
BONSAI_KEYS = {
    "architecture": "general.architecture",
    "hidden_size": "bonsai-dit.hidden_size",
    "intermediate_size": "bonsai-dit.intermediate_size",
    "num_layers": "bonsai-dit.num_layers",
    "num_attention_heads": "bonsai-dit.num_attention_heads",
    "num_key_value_heads": "bonsai-dit.num_key_value_heads",
    "latent_channels": "bonsai-dit.latent_channels",
    "patch_size": "bonsai-dit.patch_size",
    "timestep_embedding_dim": "bonsai-dit.timestep_embedding_dim",
    "quantization_type": "bonsai-dit.quantization_type",
    "block_size": "bonsai-dit.block_size",
    "scale_type": "bonsai-dit.scale_type",
    "text_encoder_hidden_size": "bonsai-dit.text_encoder_hidden_size",
    "vae_latent_channels": "bonsai-dit.vae_latent_channels",
    "default_steps": "bonsai-dit.default_steps",
    "default_guidance_scale": "bonsai-dit.default_guidance_scale",
    "scheduler": "bonsai-dit.scheduler",
}

# Bonsai 4B architecture defaults (from generate.py and model config)
BONSAI_ARCH_DEFAULTS = {
    "hidden_size": 3072,
    "intermediate_size": 8192,
    "num_layers": 24,
    "num_attention_heads": 24,
    "num_key_value_heads": 8,  # GQA
    "latent_channels": 16,
    "patch_size": 2,
    "timestep_embedding_dim": 3072,
    "text_encoder_hidden_size": 2048,
    "vae_latent_channels": 16,
    "default_steps": 4,
    "default_guidance_scale": 1.0,
    "scheduler": "euler",
}


def align(value: int, alignment: int = GGUF_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def pack_ternary(weights: np.ndarray, block_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Pack ternary weights {-1, 0, 1} into 2-bit representation with scale factors.
    
    Returns:
        packed: uint8 array of packed ternary values
        scales: float32 array of block scale factors
    """
    flat = weights.flatten().astype(np.float32)
    total_elements = len(flat)
    num_blocks = (total_elements + block_size - 1) // block_size
    
    scales = np.zeros(num_blocks, dtype=np.float32)
    packed_values = np.zeros(total_elements * 2 // 8, dtype=np.uint8)
    
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, total_elements)
        block = flat[start:end]
        
        # Compute scale as max absolute value in block
        scale = np.max(np.abs(block))
        scales[block_idx] = scale
        
        # Normalize and quantize to {-1, 0, 1}
        if scale > 0:
            normalized = block / scale
        else:
            normalized = block
            
        ternary = np.round(normalized).clip(-1, 1).astype(np.int8)
        
        # Pack to 2 bits per value: 0->-1, 1->0, 2->1
        packed_ternary = (ternary + 1).astype(np.uint8)  # 0, 1, 2
        
        # Pack into bytes (4 values per byte)
        for i in range(len(packed_ternary)):
            byte_idx = (start + i) * 2 // 8
            bit_offset = ((start + i) * 2) % 8
            packed_values[byte_idx] |= (packed_ternary[i] & 0x3) << bit_offset
    
    return packed_values, scales


def pack_binary(weights: np.ndarray, block_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Pack binary weights {-1, 1} into 1-bit representation with scale factors.
    
    Returns:
        packed: uint8 array of packed binary values
        scales: float32 array of block scale factors
    """
    flat = weights.flatten().astype(np.float32)
    total_elements = len(flat)
    num_blocks = (total_elements + block_size - 1) // block_size
    
    scales = np.zeros(num_blocks, dtype=np.float32)
    packed_values = np.zeros((total_elements + 7) // 8, dtype=np.uint8)
    
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min(start + block_size, total_elements)
        block = flat[start:end]
        
        # Compute scale
        scale = np.max(np.abs(block))
        scales[block_idx] = scale
        
        # Binary quantization: {-1, 1}
        if scale > 0:
            normalized = block / scale
        else:
            normalized = block
            
        binary = (np.round(normalized).clip(-1, 1) > 0).astype(np.uint8)  # 0 or 1
        
        # Pack into bytes (8 values per byte)
        for i in range(len(binary)):
            byte_idx = (start + i) // 8
            bit_offset = (start + i) % 8
            packed_values[byte_idx] |= binary[i] << bit_offset
    
    return packed_values, scales


def load_weights_from_dir(model_dir: Path, component: str = "transformer") -> dict[str, np.ndarray]:
    """Load weights from a model directory.
    
    Supports:
    - .npz files (numpy compressed)
    - .npy files (numpy arrays)
    - .safetensors files (via safetensors library if available)
    - Gemlite packed format (HQQ blocks)
    """
    weights = {}
    
    # Try safetensors first
    safetensors_files = list(model_dir.glob("*.safetensors"))
    if safetensors_files:
        try:
            from safetensors import safe_open
            for sf in safetensors_files:
                with safe_open(sf, framework="numpy") as f:
                    for key in f.keys():
                        weights[key] = f.get_tensor(key)
            return weights
        except ImportError:
            log.warning("safetensors not installed, skipping .safetensors files")
    
    # Try .npz files
    npz_files = list(model_dir.glob("*.npz"))
    for npz_file in npz_files:
        data = np.load(npz_file)
        for key in data.files:
            weights[key] = data[key]
    
    # Try .npy files
    npy_files = list(model_dir.glob("*.npy"))
    for npy_file in npy_files:
        key = npy_file.stem
        weights[key] = np.load(npy_file)
    
    # Try gemlite packed format (HQQ blocks)
    if not weights:
        log.info("Trying gemlite/HQQ packed format...")
        weights = load_gemlite_weights(model_dir)
    
    return weights


def load_gemlite_weights(model_dir: Path) -> dict[str, np.ndarray]:
    """Load gemlite/HQQ packed weights.
    
    Gemlite stores weights as:
    - hqq_qdata: packed quantized data
    - hqq_qscale: scale factors
    - hqq_qzero: zero points
    """
    weights = {}
    
    # Look for packed weight files
    for npz_file in model_dir.glob("*.npz"):
        data = np.load(npz_file)
        for key in data.files:
            weights[f"{npz_file.stem}/{key}"] = data[key]
    
    return weights


def write_gguf_manual(
    filepath: Path,
    weights: dict[str, np.ndarray],
    metadata: dict[str, any],
    quantization: str = "ternary",
):
    """Write GGUF file manually (fallback when gguf-py is not available)."""
    
    log.info("Writing GGUF file to %s", filepath)
    
    # Prepare tensor data
    tensors = []
    for name, array in weights.items():
        # Convert to appropriate format based on quantization
        if quantization == "ternary":
            packed, scales = pack_ternary(array)
            tensors.append((name, array.shape, GgufTensorType.TQ1_0, packed, scales))
        elif quantization == "binary":
            packed, scales = pack_binary(array)
            tensors.append((name, array.shape, GgufTensorType.IQ1_S, packed, scales))
        else:
            # Store as F32
            tensors.append((name, array.shape, GgufTensorType.F32, array.tobytes(), None))
    
    # Calculate file layout
    # Header: magic (4) + version (4) + tensor_count (8) + metadata_count (8) = 24 bytes
    # Metadata: variable
    # Tensor info: variable
    # Tensor data: variable
    
    with open(filepath, "wb") as f:
        # Write header
        f.write(GGUF_MAGIC)
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", len(metadata)))
        
        # Write metadata
        for key, value in metadata.items():
            key_bytes = key.encode("utf-8")
            f.write(struct.pack("<I", len(key_bytes)))
            f.write(key_bytes)
            
            if isinstance(value, str):
                f.write(struct.pack("<I", GgufMetadataType.STRING))
                val_bytes = value.encode("utf-8")
                f.write(struct.pack("<I", len(val_bytes)))
                f.write(val_bytes)
            elif isinstance(value, (int, np.integer)):
                f.write(struct.pack("<I", GgufMetadataType.INT32))
                f.write(struct.pack("<i", int(value)))
            elif isinstance(value, (float, np.floating)):
                f.write(struct.pack("<I", GgufMetadataType.FLOAT32))
                f.write(struct.pack("<f", float(value)))
            elif isinstance(value, bool):
                f.write(struct.pack("<I", GgufMetadataType.BOOL))
                f.write(struct.pack("<B", 1 if value else 0))
            elif isinstance(value, (list, tuple)):
                f.write(struct.pack("<I", GgufMetadataType.ARRAY))
                # Simplified array handling
                f.write(struct.pack("<I", GgufMetadataType.INT32))
                f.write(struct.pack("<I", len(value)))
                for item in value:
                    f.write(struct.pack("<i", int(item)))
        
        # Write tensor info
        for name, shape, dtype, _, _ in tensors:
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", len(shape)))
            for dim in shape:
                f.write(struct.pack("<Q", dim))
            f.write(struct.pack("<I", dtype))
            # Offset will be filled later
            f.write(struct.pack("<Q", 0))
        
        # Write tensor data
        for name, shape, dtype, data, scales in tensors:
            if scales is not None:
                # Write scales first, then packed data
                f.write(scales.tobytes())
                f.write(data.tobytes())
            else:
                f.write(data)


def write_gguf_with_library(
    filepath: Path,
    weights: dict[str, np.ndarray],
    metadata: dict[str, any],
    quantization: str = "ternary",
):
    """Write GGUF file using gguf-py library."""
    
    if not HAS_GGUF:
        raise ImportError("gguf-py not available")
    
    log.info("Writing GGUF file to %s using gguf-py", filepath)
    
    gguf_writer = gguf.GGUFBuilder()
    
    # Set metadata
    for key, value in metadata.items():
        if isinstance(value, str):
            gguf_writer.add_string(key, value)
        elif isinstance(value, (int, np.integer)):
            gguf_writer.add_int32(key, int(value))
        elif isinstance(value, (float, np.floating)):
            gguf_writer.add_float32(key, float(value))
        elif isinstance(value, bool):
            gguf_writer.add_bool(key, value)
        elif isinstance(value, (list, tuple)):
            gguf_writer.add_array(key, value)
    
    # Add tensors
    for name, array in weights.items():
        # Quantize if needed
        if quantization == "ternary":
            # TQ1_0 may not be supported, fallback to Q8_0
            try:
                gguf_writer.add_tensor(name, array, gguf.GGMLQuantizationType.TQ1_0)
            except (AttributeError, ValueError):
                log.warning("TQ1_0 not supported, using F32 for %s", name)
                gguf_writer.add_tensor(name, array.astype(np.float32))
        elif quantization == "binary":
            try:
                gguf_writer.add_tensor(name, array, gguf.GGMLQuantizationType.IQ1_S)
            except (AttributeError, ValueError):
                log.warning("IQ1_S not supported, using F32 for %s", name)
                gguf_writer.add_tensor(name, array.astype(np.float32))
        else:
            gguf_writer.add_tensor(name, array.astype(np.float32))
    
    gguf_writer.write_file_to_disk(filepath)


def get_metadata(variant: str, arch_defaults: dict | None = None) -> dict[str, any]:
    """Get GGUF metadata for Bonsai model."""
    defaults = arch_defaults or BONSAI_ARCH_DEFAULTS
    
    metadata = {
        BONSAI_KEYS["architecture"]: "bonsai-dit",
        BONSAI_KEYS["hidden_size"]: defaults["hidden_size"],
        BONSAI_KEYS["intermediate_size"]: defaults["intermediate_size"],
        BONSAI_KEYS["num_layers"]: defaults["num_layers"],
        BONSAI_KEYS["num_attention_heads"]: defaults["num_attention_heads"],
        BONSAI_KEYS["num_key_value_heads"]: defaults["num_key_value_heads"],
        BONSAI_KEYS["latent_channels"]: defaults["latent_channels"],
        BONSAI_KEYS["patch_size"]: defaults["patch_size"],
        BONSAI_KEYS["timestep_embedding_dim"]: defaults["timestep_embedding_dim"],
        BONSAI_KEYS["quantization_type"]: variant,
        BONSAI_KEYS["block_size"]: 256,
        BONSAI_KEYS["scale_type"]: GgufTensorType.F32,
        BONSAI_KEYS["text_encoder_hidden_size"]: defaults["text_encoder_hidden_size"],
        BONSAI_KEYS["vae_latent_channels"]: defaults["vae_latent_channels"],
        BONSAI_KEYS["default_steps"]: defaults["default_steps"],
        BONSAI_KEYS["default_guidance_scale"]: defaults["default_guidance_scale"],
        BONSAI_KEYS["scheduler"]: defaults["scheduler"],
    }
    
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Bonsai 4B weights to GGUF format.",
    )
    parser.add_argument(
        "--variant",
        choices=["ternary", "binary"],
        default="ternary",
        help="Model variant (default: ternary).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output GGUF file path (default: models/bonsai-{variant}.gguf).",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split output into separate GGUF files for DiT and VAE.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Model directory (default: auto-detected from variant).",
    )
    parser.add_argument(
        "--force-f32",
        action="store_true",
        help="Force F32 output (no quantization).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Resolve model directory
    if args.model_dir:
        model_dir = args.model_dir
    else:
        variant_suffix = "ternary" if args.variant == "ternary" else "binary"
        model_dir = MODELS_DIR / f"bonsai-image-4B-{variant_suffix}-gemlite"
    
    if not model_dir.exists():
        log.error("Model directory not found: %s", model_dir)
        log.error("Run: ./scripts/download_model.sh --model %s-gemlite", args.variant)
        sys.exit(1)
    
    # Load weights
    log.info("Loading weights from %s", model_dir)
    
    # Try to load transformer weights
    transformer_dir = model_dir / "transformer-gemlite-int2" if args.variant == "ternary" else model_dir / "transformer-gemlite-int1"
    if not transformer_dir.exists():
        # Fallback to any transformer directory
        transformer_dirs = list(model_dir.glob("transformer*"))
        if transformer_dirs:
            transformer_dir = transformer_dirs[0]
        else:
            transformer_dir = model_dir
    
    transformer_weights = load_weights_from_dir(transformer_dir, "transformer")
    log.info("Loaded %d transformer tensors", len(transformer_weights))
    
    # Try to load VAE weights
    vae_dir = model_dir / "vae"
    vae_weights = {}
    if vae_dir.exists():
        vae_weights = load_weights_from_dir(vae_dir, "vae")
        log.info("Loaded %d VAE tensors", len(vae_weights))
    
    # Get metadata
    metadata = get_metadata(args.variant)
    
    # Quantization type
    quantization = "f32" if args.force_f32 else args.variant
    
    # Write GGUF file(s)
    if args.split:
        # Split output
        output_dir = MODELS_DIR / "gguf"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Transformer GGUF
        transformer_gguf = output_dir / f"bonsai-{args.variant}-transformer.gguf"
        log.info("Writing transformer GGUF: %s", transformer_gguf)
        
        if HAS_GGUF:
            write_gguf_with_library(transformer_gguf, transformer_weights, metadata, quantization)
        else:
            write_gguf_manual(transformer_gguf, transformer_weights, metadata, quantization)
        
        # VAE GGUF
        if vae_weights:
            vae_gguf = output_dir / f"bonsai-{args.variant}-vae.gguf"
            log.info("Writing VAE GGUF: %s", vae_gguf)
            
            vae_metadata = {**metadata}
            vae_metadata[BONSAI_KEYS["architecture"]] = "bonsai-vae"
            
            if HAS_GGUF:
                write_gguf_with_library(vae_gguf, vae_weights, vae_metadata, quantization)
            else:
                write_gguf_manual(vae_gguf, vae_weights, vae_metadata, quantization)
    else:
        # Single file output
        output = args.output or (MODELS_DIR / f"bonsai-{args.variant}.gguf")
        output.parent.mkdir(parents=True, exist_ok=True)
        
        all_weights = {**transformer_weights, **vae_weights}
        log.info("Writing combined GGUF: %s (%d tensors)", output, len(all_weights))
        
        if HAS_GGUF:
            write_gguf_with_library(output, all_weights, metadata, quantization)
        else:
            write_gguf_manual(output, all_weights, metadata, quantization)
    
    log.info("Done!")


if __name__ == "__main__":
    main()
