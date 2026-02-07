//! Naive xor implementation
const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;

const XorModel = struct {
    or_w1: f32,
    or_w2: f32,
    or_b: f32,
    nand_w1: f32,
    nand_w2: f32,
    nand_b: f32,
    and_w1: f32,
    and_w2: f32,
    and_b: f32,

    fn init(seed: u32) XorModel {
        var prng: std.Random.DefaultPrng = .init(seed);
        const random = prng.random();

        return .{
            .or_w1 = random.float(f32),
            .or_w2 = random.float(f32),
            .or_b = random.float(f32),
            .nand_w1 = random.float(f32),
            .nand_w2 = random.float(f32),
            .nand_b = random.float(f32),
            .and_w1 = random.float(f32),
            .and_w2 = random.float(f32),
            .and_b = random.float(f32),
        };
    }

    fn forward(xor: XorModel, x1: f32, x2: f32) f32 {
        const or_x1 = sigmoid(xor.or_w1 * x1 + xor.or_w2 * x2 + xor.or_b);
        const nand_x2 = sigmoid(xor.nand_w1 * x1 + xor.nand_w2 * x2 + xor.nand_b);

        return sigmoid(xor.and_w1 * or_x1 + xor.and_w2 * nand_x2 + xor.and_b);
    }
};

const or_gate = [_][3]f32{
    .{ 0, 0, 0 },
    .{ 0, 1, 1 },
    .{ 1, 0, 1 },
    .{ 1, 1, 1 },
};

const and_gate = [_][3]f32{
    .{ 0, 0, 0 },
    .{ 0, 1, 0 },
    .{ 1, 0, 0 },
    .{ 1, 1, 1 },
};

const nand_gate = [_][3]f32{
    .{ 0, 0, 1 },
    .{ 0, 1, 1 },
    .{ 1, 0, 1 },
    .{ 1, 1, 0 },
};

const xor_gate = [_][3]f32{
    .{ 0, 0, 0 },
    .{ 0, 1, 1 },
    .{ 1, 0, 1 },
    .{ 1, 1, 0 },
};

const train_data = &xor_gate;

pub fn main() !void {
    const eps: f32 = 1e-1;
    const learning_rate: f32 = 1e-1;
    const train_count = 100_000;
    const now = try std.time.Instant.now();

    var xor: XorModel = .init(@intCast(now.timestamp.nsec));
    std.debug.print("Before: cost = {d}\n", .{cost(xor)});
    for (0..train_count) |_| {
        const gradient: XorModel = compute_gradient(&xor, eps);
        xor = apply_gradient(xor, gradient, learning_rate);
    }
    std.debug.print("After: cost = {d}\n", .{cost(xor)});

    for (train_data) |x| {
        std.debug.print(
            "{d} | {d} = {d} => {d}\n",
            .{
                x[0],
                x[1],
                x[2],
                xor.forward(x[0], x[1]),
            },
        );
    }

    std.debug.print("\n\"OR\" neuron\n", .{});
    for (train_data) |x| {
        std.debug.print(
            "{d} | {d} = {d}\n",
            .{
                x[0],
                x[1],
                sigmoid(xor.or_w1 * x[0] + xor.or_w2 * x[1] + xor.or_b),
            },
        );
    }
}

fn compute_gradient(xorp: *XorModel, eps: f32) XorModel {
    var gradient: XorModel = undefined;
    var saved: f32 = undefined;
    var xor = xorp.*;
    const c = cost(xor);

    saved = xor.or_w1;
    xor.or_w1 += eps;
    gradient.or_w1 = (cost(xor) - c) / eps;
    xor.or_w1 = saved;

    saved = xor.or_w2;
    xor.or_w2 += eps;
    gradient.or_w2 = (cost(xor) - c) / eps;
    xor.or_w2 = saved;

    saved = xor.or_b;
    xor.or_b += eps;
    gradient.or_b = (cost(xor) - c) / eps;
    xor.or_b = saved;

    saved = xor.nand_w1;
    xor.nand_w1 += eps;
    gradient.nand_w1 = (cost(xor) - c) / eps;
    xor.nand_w1 = saved;

    saved = xor.nand_w2;
    xor.nand_w2 += eps;
    gradient.nand_w2 = (cost(xor) - c) / eps;
    xor.nand_w2 = saved;

    saved = xor.nand_b;
    xor.nand_b += eps;
    gradient.nand_b = (cost(xor) - c) / eps;
    xor.nand_b = saved;

    saved = xor.and_w1;
    xor.and_w1 += eps;
    gradient.and_w1 = (cost(xor) - c) / eps;
    xor.and_w1 = saved;

    saved = xor.and_w2;
    xor.and_w2 += eps;
    gradient.and_w2 = (cost(xor) - c) / eps;
    xor.and_w2 = saved;

    saved = xor.and_b;
    xor.and_b += eps;
    gradient.and_b = (cost(xor) - c) / eps;
    xor.and_b = saved;

    return gradient;
}

fn apply_gradient(xor: XorModel, gradient: XorModel, learning_rate: f32) XorModel {
    var result = xor;
    result.or_w1 -= learning_rate * gradient.or_w1;
    result.or_w2 -= learning_rate * gradient.or_w2;
    result.or_b -= learning_rate * gradient.or_b;
    result.nand_w1 -= learning_rate * gradient.nand_w1;
    result.nand_w2 -= learning_rate * gradient.nand_w2;
    result.nand_b -= learning_rate * gradient.nand_b;
    result.and_w1 -= learning_rate * gradient.and_w1;
    result.and_w2 -= learning_rate * gradient.and_w2;
    result.and_b -= learning_rate * gradient.and_b;
    return result;
}

fn cost(xor: XorModel) f32 {
    var mean_squared_error: f32 = 0;
    for (train_data) |data| {
        const actual = data[2];
        const x1 = data[0];
        const x2 = data[1];

        const y = xor.forward(x1, x2);
        const err = y - actual;

        mean_squared_error += err * err;
    }
    return mean_squared_error / train_data.len;
}

fn random_xor() XorModel {
    var prng: std.Random.DefaultPrng = .init(testing.random_seed);
    const random = prng.random();

    return .{
        .or_w1 = random.float(f32),
        .or_w2 = random.float(f32),
        .or_b = random.float(f32),
        .nand_w1 = random.float(f32),
        .nand_w2 = random.float(f32),
        .nand_b = random.float(f32),
        .and_w1 = random.float(f32),
        .and_w2 = random.float(f32),
        .and_b = random.float(f32),
    };
}

fn sigmoid(x: f32) f32 {
    return 1 / (1 + std.math.exp(-x));
}
