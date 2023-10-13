use glam::{Mat4, Vec3, Vec4};
use num_traits::identities::Zero;

#[derive(Debug)]
pub struct ImgTransform {
    pub center: (f32, f32),
    pub rotation_x: f32,
    pub rotation_y: f32,
    pub rotation_z: f32,
    pub scale: Vec3,
    pub translation: Vec3,
}

impl Default for ImgTransform {
    fn default() -> Self {
        Self {
            scale: Vec3::ONE,
            center: Default::default(),
            rotation_x: Default::default(),
            rotation_y: Default::default(),
            rotation_z: Default::default(),
            translation: Default::default(),
        }
    }
}

impl ImgTransform {
    pub fn center(mut self, center: (f32, f32)) -> Self {
        self.center = center;
        self
    }

    pub fn rotation_x(mut self, rotation_x: f32) -> Self {
        self.rotation_x = rotation_x;
        self
    }

    pub fn rotation_y(mut self, rotation_y: f32) -> Self {
        self.rotation_y = rotation_y;
        self
    }

    pub fn rotation_z(mut self, rotation_z: f32) -> Self {
        self.rotation_z = rotation_z;
        self
    }

    pub fn scale(mut self, scale: Vec3) -> Self {
        self.scale = scale;
        self
    }

    pub fn translation(mut self, translation: Vec3) -> Self {
        self.translation = translation;
        self
    }
}

impl ImgTransform {
    pub fn construct_transform_mat(self) -> Mat4 {
        let Self {
            center: (cx, cy),
            rotation_x,
            rotation_y,
            rotation_z,
            translation,
            scale,
        } = self;

        let before = Mat4::from_translation(Vec3::new(-cx, -cy, 0.));
        let after = Mat4::from_translation(Vec3::new(cx, cy, 0.));

        let rotation_x = Mat4::from_rotation_x(rotation_x);
        let rotation_y = Mat4::from_rotation_y(rotation_y);
        let rotation_z = Mat4::from_rotation_z(rotation_z);
        let scale = Mat4::from_scale(scale);
        let translation = Mat4::from_translation(translation);

        after * translation * scale * rotation_z * rotation_y * rotation_x * before
    }
}

pub fn transform_buffer<T: Copy + Zero>(
    buffer_in: &[T],
    size: (usize, usize),
    transform: Mat4,
) -> Vec<T> {
    let mut buffer_out = vec![T::zero(); buffer_in.len()];
    transform_buffer_into(buffer_in, &mut buffer_out, size, transform);
    buffer_out
}

pub fn transform_buffer_into<T: Copy + Zero>(
    buffer_in: &[T],
    buffer_out: &mut [T],
    size: (usize, usize),
    transform: Mat4,
) {
    let (width, height) = size;
    for h in 0..height {
        for w in 0..width {
            let i = width * h + w;

            let vertex = Vec4::new(w as f32, h as f32, 0., 1.);
            let vertex = transform * vertex;
            let (w_pre, h_pre) = (vertex.x.round(), vertex.y.round());

            if (0f32..height as f32).contains(&h_pre) && (0f32..width as f32).contains(&w_pre) {
                let i_pre = (width as f32 * h_pre + w_pre) as usize;
                buffer_out[i] = buffer_in[i_pre];
            } else {
                buffer_out[i] = T::zero();
            }
        }
    }
}
