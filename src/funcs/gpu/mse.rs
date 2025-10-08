use super::{GpuFunction, KernelCall, SharedBuffer};
use crate::graph::TensorId;

pub fn gpu_impl(out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
    let predicted_shape = &inps[0];
    let target_shape = &inps[1];

    // Ensure shapes match
    assert_eq!(predicted_shape, target_shape);

    let size = predicted_shape.iter().fold(1, |acc, &dim| acc * dim);

    let forward_source = format!(
        "
        __kernel void mse_forward(__global float *output,
                                  __global float *predicted,
                                  __global float *target,
                                  uint n) {{
            uint id = get_global_id(0);
            if (id == 0) {{
                float sum = 0.0f;
                for (uint i = 0; i < n; i++) {{
                    float diff = predicted[i] - target[i];
                    sum += diff * diff;
                }}
                output[0] = sum / (float)n;
            }}
        }}"
    );

    let backward_source = format!(
        "
        __kernel void mse_backward(__global float *predicted_grad,
                                   __global float *target_grad,
                                   __global float *predicted,
                                   __global float *target,
                                   __global float *output_grad,
                                   uint n) {{
            uint id = get_global_id(0);
            if (id < n) {{
                float diff = predicted[id] - target[id];
                float grad_scale = output_grad[0] * 2.0f / (float)n;
                predicted_grad[id] = diff * grad_scale;
                target_grad[id] = -diff * grad_scale;
            }}
        }}"
    );

    GpuFunction {
        shared_buffers: vec![], // No shared buffers needed for MSE
        forward_funcs: vec![KernelCall {
            source_code: forward_source,
            kernel_name: "mse_forward".to_string(),
            global_work_size: 1, // Only one thread needed for forward pass
            local_work_size: 1,
        }],
        backward_funcs: vec![KernelCall {
            source_code: backward_source,
            kernel_name: "mse_backward".to_string(),
            global_work_size: size,
            local_work_size: 32,
        }],
    }
}
