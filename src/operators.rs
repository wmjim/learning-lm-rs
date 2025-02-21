use core::slice;

use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

/// 对输入张量 `x` 执行均方根归一化（RMS Norm）操作，并将结果存储在张量 `y` 中。
///
/// 均方根归一化公式为：`y = (x / rms(x)) * w`，其中 `rms(x)` 是 `x` 的均方根值，`w` 是可学习的权重参数。
///
/// # 参数
/// - `y`: 可变的 `Tensor<f32>` 引用，用于存储归一化后的结果。
/// - `x`: 不可变的 `Tensor<f32>` 引用，作为输入数据。
/// - `w`: 不可变的 `Tensor<f32>` 引用，作为可学习的权重参数。
/// - `epsilon`: 一个小的常量，用于避免除零错误。
pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    let len = y.size();
    assert!(len == x.size());
    let w_size = w.size();
    // 以 w_size 为步长遍历输入张量 x
    for start in (0..len).step_by(w_size) {
        // 确保不会越界访问 x 和 y 张量
        if start + w_size > len {
            break;
        }

        // 从 y 张量中切出大小为 w.shape() 的子张量
        let mut y_slice = y.slice(start, w.shape());
        // 从 x 张量中切出大小为 w.shape() 的子张量
        let x_slice = x.slice(start, w.shape());
        // 以不安全的方式获取 y_slice 数据的可变切片
        let _y = unsafe { y_slice.data_mut() };
        // 获取 x_slice 数据的不可变切片
        let _x: &[f32] = x_slice.data();
        // 获取权重张量 w 数据的不可变切片
        let _w = w.data();

        // 计算 x_slice 中元素的均方根（RMS）值
        let rms = (_x.iter().map(|&v| v * v).sum::<f32>() / _x.len() as f32 + epsilon).sqrt();
        for (y_i, (&x_ij, &w_j)) in _y.iter_mut().zip(_x.iter().zip(_w.iter())) {
            // 对 x_ij 进行归一化处理，然后乘以权重 w_j，得到归一化后的结果
            *y_i = (x_ij / rms) * w_j;
        }
    }
}

/// 实现 SwiGLU 激活函数，即 y = silu(x) * y，这是一个逐元素操作
///
/// # 参数
/// - `y`: 可变的 `Tensor<f32>` 引用，用于存储计算结果，会被修改。
/// - `x`: 不可变的 `Tensor<f32>` 引用，作为输入数据。
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    // 断言 y 和 x 张量的元素数量相同，确保 element-wise 操作不会越界
    assert!(len == x.size());

    // 以不安全的方式获取 y 张量数据的可变切片
    let _y = unsafe { y.data_mut() };
    let _x = x.data();

    for (mut_num_y, num_x) in _y.iter_mut().zip(_x.iter()) {
        // 计算 x 元素的 SiLU 激活值
        let silu_x = num_x * (1. / (1. + (-num_x).exp()));
        // 将 y 元素乘以对应的 SiLU 激活值
        *mut_num_y *= silu_x;
    }
}

/// 执行矩阵乘法 C = beta * C + alpha * A @ B^T，其中 B^T 是 B 的转置
/// 注意：这里不需要显式地对 B 进行转置操作
///
/// # 参数
/// - `c`: 可变的 `Tensor<f32>` 引用，存储最终结果，会被修改
/// - `beta`: 浮点数，用于与原矩阵 C 中的元素相乘
/// - `a`: 不可变的 `Tensor<f32>` 引用，矩阵乘法中的矩阵 A
/// - `b`: 不可变的 `Tensor<f32>` 引用，矩阵乘法中的矩阵 B
/// - `alpha`: 浮点数，用于与矩阵乘法结果 A @ B^T 中的元素相乘
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    // 获取矩阵 A 的列数，同时也是矩阵 B 的列数
    let step: usize = a.shape()[1];
    // 确保矩阵 A 和矩阵 B 的列数相同，以保证矩阵乘法可以进行
    assert!(step == b.shape()[1]);
    // 获取矩阵 A 的元素总数
    let len_a = a.size();
    // 获取矩阵 B 的元素总数
    let len_b = b.size();
    // 创建一个空的动态数组，用于存储矩阵乘法 A @ B^T 的结果
    let mut ab = vec![];
    // 以不安全的方式获取矩阵 C 数据的可变切片，以便后续修改其元素
    let _c = unsafe { c.data_mut() };

    // 外层循环遍历矩阵 A 的每一行
    for start in (0..len_a).step_by(step) {
        // 确保不会越界访问矩阵 A
        if start + step > len_a {
            break;
        }
        // 定义切片的形状
        let new_shape = vec![step];
        // 从矩阵 A 中切出大小为 step 的子张量
        let a_slice = a.slice(start, &new_shape);

        // 内层循环遍历矩阵 B 的每一行
        for start in (0..len_b).step_by(step) { 
            // 确保不会越界访问矩阵 B
            if start + step > len_b {
                break;
            }
            // 从矩阵 B 中切出大小为 step 的子张量
            let b_slice = b.slice(start, &new_shape);
            // 计算 a_slice 和 b_slice 的点积，并将结果存储在 ab 数组中
            ab.push(dot(&a_slice, &b_slice));
        }
    }
    // 遍历矩阵 C 的元素和矩阵乘法结果 ab 的元素
    for (c_i, ab_i) in _c.iter_mut().zip(ab.iter()) {
        // 根据公式 C = beta * C + alpha * A @ B^T 更新矩阵 C 的元素
        *c_i = alpha * ab_i + beta * *c_i;
    }
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    swiglu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    y.print();
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}
