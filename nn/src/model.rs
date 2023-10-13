use candle::Tensor;
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Dropout, Linear, VarBuilder};

use crate::error::NnError;

pub struct ConvNet {
    conv1: Conv2d,
    fc1: Linear,
}

impl ConvNet {
    pub fn new(vs: VarBuilder) -> Result<Self, NnError> {
        let conv1 = conv2d(
            1,
            1,
            3,
            Conv2dConfig {
                stride: 3,
                ..Default::default()
            },
            vs.pp("conv1"),
        )?;
        let fc1 = linear(81, 16, vs.pp("fc1"))?;
        Ok(Self { conv1, fc1 })
    }
    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor, NnError> {
        let xs = xs
            .reshape(((), 1, 28, 28))?
            .apply(&self.conv1)?
            .flatten_from(1)?
            .apply(&self.fc1)?
            .relu()?;
        Ok(xs)
    }
}

// pub struct ConvNet {
//     conv1: Conv2d,
//     conv2: Conv2d,
//     fc1: Linear,
//     fc2: Linear,
//     dropout: Dropout,
// }

// impl ConvNet {
//     pub fn new(vs: VarBuilder) -> Result<Self, NnError> {
//         let conv1 = conv2d(1, 32, 5, Default::default(), vs.pp("conv1"))?;
//         let conv2 = conv2d(32, 64, 5, Default::default(), vs.pp("conv2"))?;
//         let fc1 = linear(1024, 1024, vs.pp("fc1"))?;
//         let fc2 = linear(1024, 16, vs.pp("fc2"))?;
//         let dropout = Dropout::new(0.5);
//         Ok(Self {
//             conv1,
//             conv2,
//             fc1,
//             fc2,
//             dropout,
//         })
//     }
//     pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor, NnError> {
//         let xs = xs
//             .reshape(((), 1, 28, 28))?
//             .apply(&self.conv1)?
//             .max_pool2d(2)?
//             .apply(&self.conv2)?
//             .max_pool2d(2)?
//             .flatten_from(1)?
//             .apply(&self.fc1)?
//             .relu()?;
//         Ok(self.dropout.forward(&xs, train)?.apply(&self.fc2)?)
//     }
// }
