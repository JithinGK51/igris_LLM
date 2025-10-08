use super::Function;
use crate::tensor::*;

#[cfg(feature = "gpu")]
use super::{gpu, GpuFunction, TensorId};

use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct MSE {
    diff: Arc<Tensor<f32>>,
}

impl MSE {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {
            diff: Arc::new(Tensor::scalar(0.)),
        })
    }
}

impl Function for MSE {
    fn run(
        &mut self,
        inps: &[&GeneralTensor],
        _training: bool,
    ) -> Result<Tensor<f32>, TensorError> {
        let predicted = inps[0].as_float()?;
        let target = inps[1].as_float()?;

        // Ensure shapes match
        if predicted.shape() != target.shape() {
            return Err(TensorError::UnexpectedShape);
        }

        // Compute difference: predicted - target
        self.diff = Arc::new((predicted - target)?);

        // Compute MSE: mean((predicted - target)^2)
        let squared_diff = (&*self.diff * &*self.diff)?;
        let mse = Tensor::scalar(squared_diff.mean());

        Ok(mse)
    }

    fn grad(
        &self,
        inps: &[&GeneralTensor],
        out_grad: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>, TensorError> {
        let predicted = inps[0].as_float()?;
        let target = inps[1].as_float()?;

        // Gradient of MSE with respect to predicted: 2 * (predicted - target) / n
        let n = predicted.size() as f32;
        let grad_scale = out_grad.scalar()? * 2.0 / n;
        
        let grad_predicted = (&*self.diff * &Tensor::scalar(grad_scale))?;
        let grad_target = (&*self.diff * &Tensor::scalar(-grad_scale))?;

        Ok(vec![grad_predicted, grad_target])
    }

    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }

    #[cfg(feature = "gpu")]
    fn gpu_impl(&self, out_id: TensorId, inps: &[Vec<usize>]) -> GpuFunction {
        gpu::mse::gpu_impl(out_id, inps)
    }
}
