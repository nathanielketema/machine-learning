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
const symb: u8 = '^';

pub fn main(_: std.process.Init) !void {
    var prng: std.Random.DefaultPrng = .init(67);
    const random: std.Random = prng.random();

    const h: f32 = 1e-3;
    const alpha: f32 = 1;
    const epoch: usize = 50000;

    var wght1: f32 = random.float(f32);
    var wght2: f32 = random.float(f32);
    var bias: f32 = random.float(f32);

    for (0..epoch) |_| {
        // Finite difference:
        // - f(x) = (f(x + h) - f(x)) / h
        const loss: f32 = cost(wght1, wght2, bias);

        const wght1_grad: f32 = (cost(wght1 + h, wght2, bias) - loss) / h;
        const wght2_grad: f32 = (cost(wght1, wght2 + h, bias) - loss) / h;
        const bias_grad: f32 = (cost(wght1, wght2, bias + h) - loss) / h;

        wght1 -= wght1_grad * alpha;
        wght2 -= wght2_grad * alpha;
        bias -= bias_grad * alpha;
    }

    std.debug.print("Results\n", .{});
    for (training_data) |data| {
        const pred: f32 = frwd(data[0], data[1], wght1, wght2, bias);
        const loss: f32 = cost(wght1, wght2, bias);
        std.debug.print(
            "{d} {c} {d} = {d}, {d}......cost = {d}\n",
            .{
                data[0],
                symb,
                data[1],
                data[2],
                pred,
                loss,
            },
        );
    }
}

fn frwd(inpt1: f32, inpt2: f32, wght1: f32, wght2: f32, bias: f32) f32 {
    return sigmoid(inpt1 * wght1 + inpt2 * wght2 + bias);
}

fn cost(wght1: f32, wght2: f32, bias: f32) f32 {
    var loss: f32 = 0;
    for (training_data) |data| {
        const pred: f32 = frwd(data[0], data[1], wght1, wght2, bias);
        const goal: f32 = data[2];

        const diff: f32 = pred - goal;

        loss += diff * diff;
    }
    return loss / training_data.len;
}

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}
