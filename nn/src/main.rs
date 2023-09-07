use anyhow::Result;
use argh::FromArgs;
use candle::{DType, Device, D};
use candle_datasets::vision::mnist;

use candle_nn::{loss, ops, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use nn::ConvNet;
use rand::{seq::SliceRandom, thread_rng};

fn main() -> Result<()> {
    let args: Args = argh::from_env();

    match args.mode {
        Mode::Train(args) => train(args),
        _ => todo!(),
    }?;

    Ok(())
}

fn train(args: TrainArgs) -> Result<()> {
    let TrainArgs {
        epochs,
        learning_rate,
        batch_size,
        save,
    } = args;

    let dev = Device::cuda_if_available(0)?;

    let m = mnist::load()?;
    let (train_labels, train_images) = (
        m.train_labels.to_dtype(DType::U32)?.to_device(&dev)?,
        m.train_images.to_device(&dev)?,
    );
    let (test_labels, test_images) = (
        m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?,
        m.test_images.to_device(&dev)?,
    );

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let model = ConvNet::new(vs.clone())?;

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: learning_rate,
            ..Default::default()
        },
    )?;

    let n_batches = train_images.dim(0)? / batch_size;
    let mut batch_idxs: Vec<_> = (0..n_batches).collect();
    for epoch in 1..epochs {
        let mut sum_loss = 0f32;
        batch_idxs.shuffle(&mut thread_rng());
        for batch_idx in batch_idxs.iter() {
            let train_images = train_images.narrow(0, batch_idx * batch_size, batch_size)?;
            let train_labels = train_labels.narrow(0, batch_idx * batch_size, batch_size)?;
            let logits = model.forward(&train_images, true)?;
            let log_sm = ops::log_softmax(&logits, D::Minus1)?;
            let loss = loss::nll(&log_sm, &train_labels)?;
            opt.backward_step(&loss)?;
            sum_loss += loss.to_scalar::<f32>()?;
        }
        let avg_loss = sum_loss / n_batches as f32;

        let test_logits = model.forward(&test_images, false)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;

        println!(
            "{epoch:4} train loss {avg_loss:8.5} test acc: {:5.2}%",
            100. * test_accuracy
        );

        varmap.save(&save)?;
    }

    Ok(())
}

/// arguments for the model
#[derive(FromArgs, Debug)]
struct Args {
    /// mode - train or pred
    #[argh(subcommand)]
    mode: Mode,
}

#[derive(FromArgs, Debug)]
#[argh(subcommand)]
enum Mode {
    Train(TrainArgs),
    Pred(PredArgs),
}

/// training parameters
#[derive(FromArgs, Debug)]
#[argh(subcommand, name = "train")]
struct TrainArgs {
    /// epochs to run
    #[argh(option, default = "1000")]
    epochs: usize,
    /// learning Rate
    #[argh(option, default = "0.001")]
    learning_rate: f64,
    /// batch size
    #[argh(option, default = "64")]
    batch_size: usize,
    /// file save path
    #[argh(option, default = r#"String::from("output/model.safetensors")"#)]
    save: String,
}

/// training parameters
#[derive(FromArgs, Debug)]
#[argh(subcommand, name = "pred")]
struct PredArgs {}
