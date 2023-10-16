use candle::Tensor;
use candle_nn::{conv2d, linear, Activation, Conv2d, Conv2dConfig, Dropout, Linear, VarBuilder};

use crate::error::NnError;

pub struct ConvNet {
    conv1: Conv2d,
    fc1: Linear,
    fc2: Linear,
    // fc3: Linear,
    // fc4: Linear,
}

impl ConvNet {
    pub fn new(vs: VarBuilder) -> Result<Self, NnError> {
        let conv1 = conv2d(
            1,
            1,
            4,
            Conv2dConfig {
                stride: 4,
                ..Default::default()
            },
            vs.pp("conv1"),
        )?;
        let fc1 = linear(49, 200, vs.pp("fc1"))?;
        let fc2 = linear(200, 16, vs.pp("fc2"))?;
        // let fc3 = linear(100, 16, vs.pp("fc3"))?;
        // let fc4 = linear(50, 16, vs.pp("fc4"))?;
        Ok(Self {
            conv1,
            fc1,
            fc2,
            // fc3,
            // fc4,
        })
    }
    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor, NnError> {
        let xs = xs
            .reshape(((), 1, 28, 28))?
            .apply(&self.conv1)?
            .flatten_from(1)?
            .apply(&self.fc1)?
            // .relu()?
            .apply(&Activation::LeakyRelu(0.1))?
            .apply(&self.fc2)?;
        // .relu()?
        // .apply(&self.fc3)?;
        // .relu()?
        // .apply(&self.fc4)?;
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
