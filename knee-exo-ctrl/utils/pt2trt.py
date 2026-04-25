import torch
import tensorrt as trt
import sys
sys.path.insert(0, "/home/exov3/Documents/Knee_CTRL/tcn_model")
from TCN_Header_Model import TCNModel

def pt_to_trt(pt_model_path, trt_engine_path, hyperparam_config, fp16_mode=False):
    # Load PyTorch model
    model = TCNModel(hyperparam_config).eval()
    state_dict = torch.load(pt_model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.cuda()

    # Export to ONNX
    onnx_path = trt_engine_path.replace('.trt', '.onnx')
    dummy_input = torch.randn(
        1,  # batch size
        hyperparam_config['input_size'],
        hyperparam_config['window_size'],
    ).cuda()

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=18,
        do_constant_folding=True,
    )
    print(f"[INFO] ONNX model saved to {onnx_path}")

    # TensorRT engine building
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX model
    if not parser.parse_from_file(onnx_path):
        print("[ERROR] Failed to parse ONNX model:")
        for i in range(parser.num_errors):
            print(f"  {parser.get_error(i)}")
        raise RuntimeError("ONNX model parsing failed.")

    # Build TensorRT engine
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    if fp16_mode:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[INFO] Building engine in FP16 mode.")
        else:
            print("[WARNING] FP16 not supported on this platform. Building in FP32 mode.")

    print("[INFO] Building TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed.")

    # Save engine to file
    with open(trt_engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"[SUCCESS] TensorRT engine saved at: {trt_engine_path}")


# Example usage
pt_model = '/home/exov3/Documents/Knee_CTRL/tcn_model/smile_0.2.19_1.pt'
trt_engine = '/home/exov3/Documents/Knee_CTRL/tcn_model/smile_0.2.19_1.trt'

hyperparam_config = {
    'wandb_project_name': 'baseline_TCN',
    'wandb_session_name': 'baseline_TCN_nature_hyperparam',
    'input_size': 2, 
    'output_size': 2, # 1 for right hip torque
    'architecture': 'TCN',
    
    'transfer_learning': False,
    'dataset_proportion': 1.0, # dataset proportion for training
    
    'epochs': 50,
    'batch_size': 64,
    'init_lr': 5e-5,
    'dropout': 0.15,
    'validation_split': 0.1,
    'number_of_workers': 10,

    'window_size'       : 95,
    'number_of_layers'  : 2,
    'batch_size'        : 256,
    'epochs'            : 100,
    'learning_rate'     : 5e-4,
    'dropout'           : 0.15,
    'num_channels'      : [80,80,80,80,80],
    'kernel_size'       : 5,
    'dilations'         : [1,2,4,8,16],
    'num_workers'       : 10,
}

pt_to_trt(pt_model, trt_engine, hyperparam_config, fp16_mode=False)
