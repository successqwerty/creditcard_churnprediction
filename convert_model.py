import json
from pathlib import Path
import numpy as np
import tensorflow as tf

def convert_h5_to_tfjs():
    BASE_DIR = Path(__file__).resolve().parent
    H5_MODEL_PATH = BASE_DIR / "model" / "model.h5"
    TFJS_OUTPUT_DIR = BASE_DIR / "frontend" / "tfjs_model"

    TFJS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading Keras model from {H5_MODEL_PATH}...")
    keras_model = tf.keras.models.load_model(H5_MODEL_PATH)

    # 1. Build weights manifest and packed binary buffer
    weights_manifest = []
    weight_specs = []
    binary_buffers = []

    # Iterate layers in model
    for layer in keras_model.layers:
        weights = layer.get_weights()
        if not weights:
            continue
        
        # Layer weight names in TF.js: <layer_name>/kernel and <layer_name>/bias
        kernel_val = weights[0].astype(np.float32)
        bias_val = weights[1].astype(np.float32)

        kernel_name = f"{layer.name}/kernel"
        bias_name = f"{layer.name}/bias"

        weight_specs.append({
            "name": kernel_name,
            "shape": list(kernel_val.shape),
            "dtype": "float32"
        })
        binary_buffers.append(kernel_val.tobytes())

        weight_specs.append({
            "name": bias_name,
            "shape": list(bias_val.shape),
            "dtype": "float32"
        })
        binary_buffers.append(bias_val.tobytes())

    bin_filename = "group1-shard1of1.bin"
    bin_path = TFJS_OUTPUT_DIR / bin_filename
    full_bin_data = b"".join(binary_buffers)

    with open(bin_path, "wb") as f:
        f.write(full_bin_data)
    print(f"Saved weight binary buffer: {bin_path} ({len(full_bin_data)} bytes)")

    # 2. Build model topology (TF.js Keras layers-model structure)
    layers_config = [
        {
            "class_name": "InputLayer",
            "config": {
                "batch_input_shape": [None, 32],
                "dtype": "float32",
                "sparse": False,
                "name": "input_layer"
            }
        }
    ]

    for layer in keras_model.layers:
        layer_cfg = {
            "class_name": "Dense",
            "config": {
                "name": layer.name,
                "trainable": True,
                "dtype": "float32",
                "units": layer.units,
                "activation": layer.activation.__name__,
                "use_bias": layer.use_bias,
                "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": None}},
                "bias_initializer": {"class_name": "Zeros", "config": {}}
            }
        }
        layers_config.append(layer_cfg)

    model_topology = {
        "class_name": "Sequential",
        "config": {
            "name": "sequential",
            "layers": layers_config
        },
        "keras_version": "2.15.0",
        "backend": "tensorflow"
    }

    tfjs_model_json = {
        "format": "layers-model",
        "generatedBy": "custom-keras-tfjs-converter",
        "convertedBy": "TensorFlow.js Converter",
        "modelTopology": model_topology,
        "weightsManifest": [
            {
                "paths": [bin_filename],
                "weights": weight_specs
            }
        ]
    }

    model_json_path = TFJS_OUTPUT_DIR / "model.json"
    with open(model_json_path, "w", encoding="utf-8") as f:
        json.dump(tfjs_model_json, f, indent=2)

    print(f"Saved TF.js model JSON: {model_json_path} ({model_json_path.stat().st_size} bytes)")
    print("--- CONVERSION SUCCESSFUL ---")

if __name__ == "__main__":
    convert_h5_to_tfjs()
