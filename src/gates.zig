const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;

const and_gate = [_][3]f32{
    .{ 0, 0, 0 },
    .{ 0, 1, 0 },
    .{ 1, 0, 0 },
    .{ 1, 1, 1 },
};

const or_gate = [_][3]f32{
    .{ 0, 0, 0 },
    .{ 0, 1, 1 },
    .{ 1, 0, 1 },
    .{ 1, 1, 1 },
};

const nand_gate = [_][3]f32{
    .{ 0, 0, 1 },
    .{ 0, 1, 1 },
    .{ 1, 0, 1 },
    .{ 1, 1, 0 },
};

const training_data = &or_gate;
const symbol: u8 = '^';

pub fn main(_: std.process.Init) !void {
    var prng: std.Random.DefaultPrng = .init(67);
    const random: std.Random = prng.random();

    const h: f32 = 1e-3;
    const alpha: f32 = 1;
    const epoch: usize = 50000;

    var w1: f32 = random.float(f32);
    var w2: f32 = random.float(f32);
    var b: f32 = random.float(f32);

    for (0..epoch) |_| {
        // Finite difference:
        // - f(x) = (f(x + h) - f(x)) / h
        const c: f32 = cost(w1, w2, b);

        const w1_grad: f32 = (cost(w1 + h, w2, b) - c) / h;
        const w2_grad: f32 = (cost(w1, w2 + h, b) - c) / h;
        const b_grad: f32 = (cost(w1, w2, b + h) - c) / h;

        w1 -= w1_grad * alpha;
        w2 -= w2_grad * alpha;
        b -= b_grad * alpha;
    }

    std.debug.print("Results\n", .{});
    for (training_data) |data| {
        const pred: f32 = forward(data[0], data[1], w1, w2, b);
        const c: f32 = cost(w1, w2, b);
        std.debug.print(
            "{d} {c} {d} = {d}, {d}......cost = {d}\n",
            .{
                data[0],
                symbol,
                data[1],
                data[2],
                pred,
                c,
            },
        );
    }
}

fn forward(input_1: f32, input_2: f32, w1: f32, w2: f32, b: f32) f32 {
    return sigmoid(input_1 * w1 + input_2 * w2 + b);
}

fn cost(w1: f32, w2: f32, b: f32) f32 {
    var mse: f32 = 0;
    for (training_data) |data| {
        const pred: f32 = forward(data[0], data[1], w1, w2, b);
        const real: f32 = data[2];

        const diff: f32 = pred - real;
        mse += diff * diff;
    }
    return mse / training_data.len;
}

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}
