mod model;

pub use model::ConvNet;

pub mod error {
    use thiserror::Error;

    #[derive(Error, Debug)]
    pub enum NnError {
        #[error(transparent)]
        CandleError(#[from] candle::error::Error),
    }
}
