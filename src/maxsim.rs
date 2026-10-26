use candle_core::Tensor;
use candle_core::CustomOp2;
use candle_core::backend::{BackendStorage, BackendDevice};
use candle_core::cuda_backend::cudarc::driver::PushKernelArg;
use candle_core::cuda_backend::WrapErr;

mod kernels {
    include!("cuda_kernels.rs");
}

pub struct MaxSim {
    document_lengths: Vec<usize>,
}

impl MaxSim {
    pub fn new(lengths: Vec<usize>) -> Self {
        Self {
            document_lengths: lengths,
        }
    }
}

impl CustomOp2 for MaxSim {
    fn name(&self) -> &'static str {
        "MaxSim"
    }

    fn cpu_fwd(
        &self,
        query: &candle_core::CpuStorage,
        query_layout: &candle_core::Layout,
        document: &candle_core::CpuStorage,
        document_layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        unreachable!()
    }

    fn cuda_fwd(
        &self,
        query: &candle_core::CudaStorage,
        query_layout: &candle_core::Layout,
        document: &candle_core::CudaStorage,
        document_layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::cuda_backend::cudarc::driver::LaunchConfig;
        let q_dims = query_layout.dims();
        let d_dims = document_layout.dims();
        if q_dims.len() != 2 {
            candle_core::bail!(
                "query must have shape [n_tokens, token_dim]. got {:?}",
                q_dims
            );
        }
        if d_dims.len() != 2 {
            candle_core::bail!(
                "document must have shape [n_tokens, token_dim]. got {:?}",
                d_dims
            );
        }

        let d_query = q_dims[q_dims.len() - 1];
        let d_document = q_dims[q_dims.len() - 1];
        if d_query != d_document {
            candle_core::bail!(
                "last dimensions of query and document must match. Got {d_query} != {d_document}."
            )
        }

        if !query.device().same_device(document.device()) {
            candle_core::bail!(
                "query and document must be on the same device. got {:?} and {:?}.",
                query.device(),
                document.device()
            );
        }
        let dev = query.device();
        let func = dev
            .get_or_load_custom_func("max_sim", "mymodule", kernels::MAXSIM)
            .expect("kernel should exist");

        let cfg = LaunchConfig {
            block_dim: (1024, 1, 1),
            grid_dim: (self.document_lengths.len() as u32, 1, 1),
            shared_mem_bytes: (q_dims[0] * std::mem::size_of::<f32>()) as u32,
        };

        let dst = unsafe { dev.alloc::<f32>(self.document_lengths.len()) }?;
        let lengths = Tensor::from_iter(
            self.document_lengths.iter().map(|x| *x as i32),
            &candle_core::Device::Cuda(query.device().clone()),
        )
        .unwrap();
        let (s, l) = lengths.storage_and_layout();
        let lengths = match *s {
            candle_core::Storage::Cuda(ref s) => s,
            _ => unreachable!(),
        };

        let mut cumsum = 0;
        let starts = Tensor::from_iter(
            self.document_lengths.iter().map(|x| {
                let ret = cumsum;
                cumsum += *x;
                ret as i32
            }),
            &candle_core::Device::Cuda(query.device().clone()),
        )
        .unwrap();
        let (s, l) = starts.storage_and_layout();
        let starts = match *s {
            candle_core::Storage::Cuda(ref s) => s,
            _ => unreachable!(),
        };

        let mut builder = func.builder();
        builder
            .arg(query.as_cuda_slice::<f32>()?)
            .arg(document.as_cuda_slice::<f32>()?)
            .arg(starts.as_cuda_slice::<i32>()?)
            .arg(lengths.as_cuda_slice::<i32>()?)
            .arg(&dst);

        candle_core::builder_arg!(builder, q_dims[0], d_document);
        unsafe { builder.launch(cfg) }.w().unwrap();
        let dst = candle_core::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, (self.document_lengths.len(),).into()))
    }
}



