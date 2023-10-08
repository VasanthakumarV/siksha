use anyhow::Result;
use candle::{safetensors::save, Device, IndexOp, Tensor};
use candle_datasets::vision::mnist;
use glam::Vec3;
use rand::{distributions::Uniform, prelude::Distribution};

use nn::transform;

fn main() -> Result<()> {
    let m = mnist::load()?;

    // let imgs = m.train_images;
    let imgs = m.test_images;

    // imgs.i(..10)?
    //     .save_safetensors("test-images", "output/test-images.safetensors")?;
    // todo!();

    let b_size = imgs.dim(0)?;
    let imgs = imgs.to_vec2::<f32>()?;

    let rng = rand::thread_rng();
    let rand_trans: Vec<f32> = Uniform::new_inclusive(-10., 10.)
        .sample_iter(rng.clone())
        .take(2 * b_size)
        .collect();
    let rand_scales: Vec<f32> = Uniform::new_inclusive(0.5, 1.2)
        .sample_iter(rng.clone())
        .take(2 * b_size)
        .collect();

    let mut output_imgs = vec![0.; b_size * 28 * 28];
    let mut inverse_transforms = vec![0.; b_size * 16];
    for (i, img) in imgs.iter().enumerate() {
        let i_strided = i * 2;
        let scale = Vec3::new(rand_scales[i_strided], rand_scales[i_strided + 1], 1.);
        let translation = Vec3::new(rand_trans[i_strided], rand_trans[i_strided + 1], 0.);
        let transform = transform::ImgTransform::default()
            .center((14., 14.))
            .scale(scale)
            .translation(translation)
            .construct_transform_mat();

        let (start, end) = (i * 784, i * 784 + 784);
        transform::transform_buffer_into(img, &mut output_imgs[start..end], (28, 28), transform);
        let (start, end) = (i * 16, i * 16 + 16);
        inverse_transforms.splice(start..end, transform.inverse().to_cols_array());
    }

    let output_imgs = Tensor::from_vec(output_imgs, (b_size, 28 * 28), &Device::Cpu)?;
    let inverse_transforms = Tensor::from_vec(inverse_transforms, (b_size, 16), &Device::Cpu)?;
    let output = [("imgs", output_imgs), ("transforms", inverse_transforms)]
        .into_iter()
        .collect();

    // save(&output, "output/train.safetensors")?;
    save(&output, "output/test.safetensors")?;

    // let inp = imgs[200].iter().map(|&i| (i * 255.) as u8).collect();
    // let image = GrayImage::from_raw(28, 28, inp).unwrap();
    // image.save("output/mnist-before.png")?;
    // let image = GrayImage::from_raw(
    //     28,
    //     28,
    //     (output_imgs.i(200)? * 255.)?
    //         .to_dtype(DType::U8)?
    //         .to_vec1::<u8>()?,
    // )
    // .unwrap();
    // image.save("output/mnist.png")?;

    Ok(())
}
