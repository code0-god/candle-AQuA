use super::*;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::scalar::Scalar;

// allow: SIZE_OK — BackendStorage requires one indivisible trait implementation.
impl BackendStorage for AquaStorage {
    type Device = AquaDevice;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        let cpu = self.cpu.try_clone(layout)?;
        Ok(self.replace_cpu_storage(cpu))
    }

    fn dtype(&self) -> DType {
        self.cpu.dtype()
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Ok(self.cpu.clone())
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        Ok(self.replace_cpu_storage(self.cpu.affine(layout, mul, add)?))
    }

    fn powf(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        Ok(self.replace_cpu_storage(self.cpu.powf(layout, alpha)?))
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        Ok(self.replace_cpu_storage(self.cpu.elu(layout, alpha)?))
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, dims: &[usize]) -> Result<Self> {
        Ok(self.replace_cpu_storage(self.cpu.reduce_op(op, layout, dims)?))
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_layout: &Layout, rhs_layout: &Layout) -> Result<Self> {
        self.ensure_same_device(rhs, "cmp")?;

        Ok(self.replace_cpu_storage(self.cpu.cmp(op, &rhs.cpu, lhs_layout, rhs_layout)?))
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        Ok(self.replace_cpu_storage(self.cpu.to_dtype(layout, dtype)?))
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        Ok(self.replace_cpu_storage(self.cpu.unary_impl::<B>(layout)?))
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        self.ensure_same_device(rhs, B::NAME)?;

        Ok(self.replace_cpu_storage(
            self.cpu
                .binary_impl::<B>(&rhs.cpu, lhs_layout, rhs_layout)?,
        ))
    }

    fn where_cond(
        &self,
        layout: &Layout,
        true_value: &Self,
        true_layout: &Layout,
        false_value: &Self,
        false_layout: &Layout,
    ) -> Result<Self> {
        self.ensure_same_device(true_value, "where")?;
        self.ensure_same_device(false_value, "where")?;

        Ok(self.replace_cpu_storage(self.cpu.where_cond(
            layout,
            &true_value.cpu,
            true_layout,
            &false_value.cpu,
            false_layout,
        )?))
    }

    fn conv1d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_layout: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        self.ensure_same_device(kernel, "conv1d")?;

        Ok(self.replace_cpu_storage(
            self.cpu
                .conv1d(layout, &kernel.cpu, kernel_layout, params)?,
        ))
    }

    fn conv_transpose1d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_layout: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        self.ensure_same_device(kernel, "conv-transpose1d")?;

        Ok(self.replace_cpu_storage(self.cpu.conv_transpose1d(
            layout,
            &kernel.cpu,
            kernel_layout,
            params,
        )?))
    }

    fn conv2d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_layout: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        self.ensure_same_device(kernel, "conv2d")?;

        Ok(self.replace_cpu_storage(
            self.cpu
                .conv2d(layout, &kernel.cpu, kernel_layout, params)?,
        ))
    }

    fn conv_transpose2d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_layout: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        self.ensure_same_device(kernel, "conv-transpose2d")?;

        Ok(self.replace_cpu_storage(self.cpu.conv_transpose2d(
            layout,
            &kernel.cpu,
            kernel_layout,
            params,
        )?))
    }

    fn avg_pool2d(
        &self,
        layout: &Layout,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        Ok(self.replace_cpu_storage(self.cpu.avg_pool2d(layout, kernel, stride)?))
    }

    fn max_pool2d(
        &self,
        layout: &Layout,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        Ok(self.replace_cpu_storage(self.cpu.max_pool2d(layout, kernel, stride)?))
    }

    fn upsample_nearest1d(&self, layout: &Layout, target: usize) -> Result<Self> {
        Ok(self.replace_cpu_storage(self.cpu.upsample_nearest1d(layout, target)?))
    }

    fn upsample_nearest2d(
        &self,
        layout: &Layout,
        target_h: usize,
        target_w: usize,
    ) -> Result<Self> {
        Ok(self.replace_cpu_storage(self.cpu.upsample_nearest2d(layout, target_h, target_w)?))
    }

    fn upsample_bilinear2d(
        &self,
        layout: &Layout,
        target_h: usize,
        target_w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        Ok(self.replace_cpu_storage(self.cpu.upsample_bilinear2d(
            layout,
            target_h,
            target_w,
            align_corners,
            scale_h,
            scale_w,
        )?))
    }

    fn gather(
        &self,
        layout: &Layout,
        indexes: &Self,
        indexes_layout: &Layout,
        dim: usize,
    ) -> Result<Self> {
        self.ensure_same_device(indexes, "gather")?;

        Ok(self.replace_cpu_storage(self.cpu.gather(layout, &indexes.cpu, indexes_layout, dim)?))
    }

    fn scatter_set(
        &mut self,
        layout: &Layout,
        indexes: &Self,
        indexes_layout: &Layout,
        source: &Self,
        source_layout: &Layout,
        dim: usize,
    ) -> Result<()> {
        self.ensure_same_device(indexes, "scatter-set")?;
        self.ensure_same_device(source, "scatter-set")?;

        self.cpu.scatter_set(
            layout,
            &indexes.cpu,
            indexes_layout,
            &source.cpu,
            source_layout,
            dim,
        )
    }

    fn scatter_add_set(
        &mut self,
        layout: &Layout,
        indexes: &Self,
        indexes_layout: &Layout,
        source: &Self,
        source_layout: &Layout,
        dim: usize,
    ) -> Result<()> {
        self.ensure_same_device(indexes, "scatter-add")?;
        self.ensure_same_device(source, "scatter-add")?;

        self.cpu.scatter_add_set(
            layout,
            &indexes.cpu,
            indexes_layout,
            &source.cpu,
            source_layout,
            dim,
        )
    }

    fn index_select(
        &self,
        indexes: &Self,
        input_layout: &Layout,
        indexes_layout: &Layout,
        dim: usize,
    ) -> Result<Self> {
        self.ensure_same_device(indexes, "index-select")?;

        Ok(self.replace_cpu_storage(self.cpu.index_select(
            &indexes.cpu,
            input_layout,
            indexes_layout,
            dim,
        )?))
    }

    fn index_add(
        &self,
        layout: &Layout,
        indexes: &Self,
        indexes_layout: &Layout,
        source: &Self,
        source_layout: &Layout,
        dim: usize,
    ) -> Result<Self> {
        self.ensure_same_device(indexes, "index-add")?;
        self.ensure_same_device(source, "index-add")?;

        Ok(self.replace_cpu_storage(self.cpu.index_add(
            layout,
            &indexes.cpu,
            indexes_layout,
            &source.cpu,
            source_layout,
            dim,
        )?))
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        self.ensure_same_device(rhs, "matmul")?;

        let request = AquaMatMulRequest {
            lhs: &self.cpu,
            rhs: &rhs.cpu,
            bmnk,
            lhs_layout,
            rhs_layout,
        };

        let cpu = match self.device.executor.matmul(request)? {
            AquaDispatch::Executed(storage) => {
                self.validate_matmul_output(&storage, bmnk)?;
                storage
            }
            AquaDispatch::Fallback => self.cpu.matmul(&rhs.cpu, bmnk, lhs_layout, rhs_layout)?,
        };

        Ok(self.replace_cpu_storage(cpu))
    }

    fn copy_strided_src(
        &self,
        dst: &mut Self,
        dst_offset: usize,
        src_layout: &Layout,
    ) -> Result<()> {
        self.ensure_same_device(dst, "copy")?;

        self.cpu
            .copy_strided_src(&mut dst.cpu, dst_offset, src_layout)
    }

    #[allow(clippy::too_many_arguments)]
    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_stride: usize,
        dst_stride: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        self.ensure_same_device(dst, "copy2d")?;

        self.cpu.copy2d(
            &mut dst.cpu,
            d1,
            d2,
            src_stride,
            dst_stride,
            src_offset,
            dst_offset,
        )
    }

    fn const_set(&mut self, value: Scalar, layout: &Layout) -> Result<()> {
        self.cpu.const_set(value, layout)
    }
}
