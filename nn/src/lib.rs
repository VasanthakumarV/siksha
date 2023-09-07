use thiserror::Error;

pub mod model;

pub use model::ConvNet;

#[derive(Error, Debug)]
pub enum NnError {
    #[error(transparent)]
    CandleError(#[from] candle::error::Error),
}
