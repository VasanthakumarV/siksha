use anyhow::Result;
use argh::FromArgs;
use candle::{safetensors::load, DType, Device, IndexOp};
use candle_nn::{loss::mse, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::{seq::SliceRandom, thread_rng};

use nn::ConvNet;

fn main() -> Result<()> {
    let args: Args = argh::from_env();

    match args.mode {
        Mode::Train(args) => train(args),
        Mode::Pred(args) => pred(args),
    }?;

    Ok(())
}

fn pred(args: PredArgs) -> Result<()> {
    let PredArgs {
        model_file,
        input_images,
    } = args;

    let dev = Device::cuda_if_available(0)?;

    let images = load(input_images, &dev)?;
    let images = images.get("imgs").unwrap();

    let mut varmap = VarMap::new();
    varmap.load(model_file)?;
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let model = ConvNet::new(vs.clone())?;

    let preds = model.forward(images, false)?;
    dbg!(preds);

    Ok(())
}

fn train(args: TrainArgs) -> Result<()> {
    let TrainArgs {
        train_data,
        test_data,
        epochs,
        learning_rate,
        batch_size,
        save,
    } = args;

    let dev = Device::cuda_if_available(0)?;

    let train = load(train_data, &dev)?;
    let test = load(test_data, &dev)?;
    let (train_images, train_targets) =
        (train.get("imgs").unwrap(), train.get("transforms").unwrap());
    let (test_images, test_targets) = (test.get("imgs").unwrap(), test.get("transforms").unwrap());

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
            let train_targets = train_targets.narrow(0, batch_idx * batch_size, batch_size)?;
            let preds = model.forward(&train_images, true)?;
            let loss = mse(&preds, &train_targets)?;
            opt.backward_step(&loss)?;
            sum_loss += loss.to_scalar::<f32>()?;
        }
        let avg_loss = sum_loss / n_batches as f32;

        let test_preds = model.forward(test_images, false)?;
        let test_loss = mse(&test_preds, test_targets)?.to_scalar::<f32>()?;
        dbg!(
            test_preds.i(0)?.to_vec1::<f32>()?,
            test_targets.i(0)?.to_vec1::<f32>()?
        );
        println!("{epoch:4} | train-loss {avg_loss:8.5} | test-loss: {test_loss:8.5}");

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
    /// training data
    #[argh(option, default = r#"String::from("output/train.safetensors")"#)]
    train_data: String,
    /// testing data
    #[argh(option, default = r#"String::from("output/test.safetensors")"#)]
    test_data: String,
    /// epochs to run
    #[argh(option, default = "1000")]
    epochs: usize,
    /// learning rate
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
struct PredArgs {
    /// model file path
    #[argh(option, default = r#"String::from("output/model.safetensors")"#)]
    model_file: String,
    /// input images file
    #[argh(option, default = r#"String::from("output/test.safetensors")"#)]
    input_images: String,
}
