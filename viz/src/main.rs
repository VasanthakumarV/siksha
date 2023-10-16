#![feature(lazy_cell)]

use std::sync::LazyLock;

use candle::{safetensors::load_buffer, DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use glam::{Mat4, Vec3};
use leptos::wasm_bindgen::JsCast;
use leptos::{html::Canvas, *};
use rand::{seq::SliceRandom, thread_rng};
use wasm_bindgen::Clamped;
use web_sys::{CanvasRenderingContext2d, ImageData};

use nn::transform::{transform_buffer, ImgTransform};
use nn::ConvNet;

static DATA: LazyLock<Tensor> = LazyLock::new(|| {
    let buffer = include_bytes!("../../output/test-images.safetensors");
    let data = load_buffer(buffer, &Device::Cpu).unwrap();
    data.get("test-images").unwrap().clone()
});

static MODEL: LazyLock<ConvNet> = LazyLock::new(|| {
    let buffer = include_bytes!("../../output/model.safetensors");
    let weights = load_buffer(buffer, &Device::Cpu).unwrap();
    let vs = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    ConvNet::new(vs).unwrap()
});

fn main() {
    console_error_panic_hook::set_once();

    mount_to_body(|| {
        view! { <App/> }
    })
}

#[component]
fn App() -> impl IntoView {
    let mut rng = thread_rng();
    let (batch_size, _) = DATA.dims2().unwrap();
    let choices = (0..batch_size).collect::<Vec<_>>();

    let idx = RwSignal::new(*choices.choose(&mut rng).unwrap());

    let on_click = move |_| {
        let choice = choices.choose(&mut rng).unwrap();
        idx.set(*choice);
    };

    let image_tensor = move || DATA.i(idx()).unwrap();

    let (scale_x, scale_y) = (RwSignal::new(1.), RwSignal::new(1.));
    let (trans_x, trans_y) = (RwSignal::new(0.), RwSignal::new(0.));

    let transform = move || {
        ImgTransform::default()
            .center((14., 14.))
            .scale(Vec3::new(scale_x(), scale_y(), 1.))
            .translation(Vec3::new(trans_x(), trans_y(), 0.))
            .construct_transform_mat()
    };
    let transformed_image = move || {
        let image_tensor = image_tensor();
        let transformed_img = transform_buffer(
            &image_tensor.to_vec1::<f32>().unwrap(),
            (28, 28),
            transform(),
        );
        Tensor::from_vec(transformed_img, image_tensor.shape(), image_tensor.device()).unwrap()
    };

    let inverse_image = move || {
        let transformed_image = transformed_image();
        let inverse_transform = MODEL
            .forward(&transformed_image.unsqueeze(0).unwrap(), false)
            .unwrap();
        let inverse_vec = inverse_transform
            .squeeze(0)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // log!("{:?}", inverse_vec);
        let mut inverse_transform = [0.; 16];
        inverse_transform
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = inverse_vec[i]);
        let inverse_transform = Mat4::from_cols_array(&inverse_transform);
        let inverse_image = transform_buffer(
            &transformed_image.to_vec1::<f32>().unwrap(),
            (28, 28),
            inverse_transform,
        );
        Tensor::from_vec(
            inverse_image,
            transformed_image.shape(),
            transformed_image.device(),
        )
        .unwrap()
    };

    fn input_view(min: f32, max: f32, rw_signal: RwSignal<f32>) -> impl IntoView {
        let scale = RwSignal::new(rw_signal.get_untracked());
        view! {
            <input
                class="range w-52"
                type="range"
                min=min
                max=max
                step=0.1
                prop:value=scale
                on:input=move |ev| { scale.set(event_target_value(&ev).parse().unwrap()) }

                on:change=move |ev| { rw_signal.set(event_target_value(&ev).parse().unwrap()) }
            />
            <pre class="text-sm my-auto -ml-3">{move || { format!("{:>4}", scale()) }}</pre>
        }
    }

    view! {
        <div class="flex flex-col gap-10 justify-center p-10 sm:flex-row">
            <div class="flex flex-col gap-y-4">
                <label>"Scale"</label>
                <div class="flex flex-row gap-x-4 justify-center">
                    <label>"x"</label>
                    {input_view(0.5, 1.2, scale_x)}
                </div>
                <div class="flex flex-row gap-x-4 justify-center">
                    <label>"y"</label>
                    {input_view(0.5, 1.2, scale_y)}
                </div>
                <label>"Translate"</label>
                <div class="flex flex-row gap-x-4 justify-center">
                    <label>"x"</label>
                    {input_view(-10., 10., trans_x)}
                </div>
                <div class="flex flex-row gap-x-4 justify-center">
                    <label>"y"</label>
                    {input_view(-10., 10., trans_y)}
                </div>
            </div>
            <div class="divider sm:divider-horizontal"></div>
            <div class="flex flex-col items-center gap-y-4">
                <label class="self-start">"Input Image"</label>
                <Image image_tensor=Signal::derive(transformed_image)/>
                <button class="btn" on:click=on_click>
                    "Randomize"
                </button>
            </div>
            <div class="divider sm:divider-horizontal"></div>
            <div class="flex flex-col items-center gap-y-4">
                <label class="self-start">"Output Image"</label>
                <Image image_tensor=Signal::derive(inverse_image)/>
            </div>
        </div>
    }
}

#[component]
fn Image(image_tensor: Signal<Tensor>) -> impl IntoView {
    let (width, height) = (100, 100);
    let node_ref = create_node_ref::<Canvas>();

    let image = move || tensor_to_image(&image_tensor(), width, height).unwrap();

    let render_image = move |canvas: HtmlElement<Canvas>, image: &Vec<u8>| {
        canvas
            .get_context("2d")
            .map(|ctx| {
                let ctx = ctx.unwrap().dyn_into::<CanvasRenderingContext2d>().unwrap();
                ImageData::new_with_u8_clamped_array_and_sh(Clamped(image), width, height)
                    .as_ref()
                    .map(|data| ctx.put_image_data(data, 0., 0.).unwrap())
                    .unwrap();
            })
            .unwrap();
    };

    node_ref.on_load(move |canvas| {
        image.into_signal().with_untracked(|image| {
            render_image(canvas, image);
        });
    });

    create_effect(move |_| {
        if let Some(canvas) = node_ref() {
            image.into_signal().with(|image| {
                render_image(canvas, image);
            });
        }
    });

    view! { <canvas _ref=node_ref width=width height=height></canvas> }
}

fn tensor_to_image(tensor: &Tensor, width: u32, height: u32) -> candle::Result<Vec<u8>> {
    let tensor = tensor
        .reshape((1, 1, 28, 28))?
        .upsample_nearest2d(width as usize, height as usize)?
        .flatten_all()?;
    let rgb = tensor.unsqueeze(1)?;
    let alpha = Tensor::ones_like(&rgb)?;
    let rgba = Tensor::cat(&[&rgb, &rgb, &rgb, &alpha], 1)?.flatten_all()?;
    (rgba * 255.)?.to_dtype(DType::U8)?.to_vec1()
}
