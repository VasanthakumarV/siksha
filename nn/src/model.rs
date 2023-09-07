use candle::Tensor;
use candle_nn::{conv2d, linear, Conv2d, Dropout, Linear, VarBuilder};

use crate::NnError;

pub struct ConvNet {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
    dropout: Dropout,
}

impl ConvNet {
    pub fn new(vs: VarBuilder) -> Result<Self, NnError> {
        let conv1 = conv2d(1, 32, 5, Default::default(), vs.pp("conv1"))?;
        let conv2 = conv2d(32, 64, 5, Default::default(), vs.pp("conv2"))?;
        let fc1 = linear(1024, 1024, vs.pp("fc1"))?;
        let fc2 = linear(1024, 10, vs.pp("fc2"))?;
        let dropout = Dropout::new(0.5);
        Ok(Self {
            conv1,
            conv2,
            fc1,
            fc2,
            dropout,
        })
    }
    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor, NnError> {
        let (b_sz, _img_dim) = xs.dims2()?;
        let xs = xs
            .reshape((b_sz, 1, 28, 28))?
            .apply(&self.conv1)?
            .max_pool2d(2)?
            .apply(&self.conv2)?
            .max_pool2d(2)?
            .flatten_from(1)?
            .apply(&self.fc1)?
            .relu()?;
        Ok(self.dropout.forward(&xs, train)?.apply(&self.fc2)?)
    }
}
