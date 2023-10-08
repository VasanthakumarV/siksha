use anyhow::Result;
use candle::{safetensors::load, Device};

struct Batcher {
    input: Tensor,
    target: Tensor,
}

impl Iterator for Batcher {
    type Item = Tensor;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

fn main() -> Result<()> {
    let data = load("output/test.safetensors", &Device::Cpu)?;
    let input = data.get("imgs").unwrap();
    dbg!(input.shape());

    Ok(())
}
