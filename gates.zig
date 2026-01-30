const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;

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

const train_data = &and_gate;

pub fn main() !void {
    var prng: std.Random.DefaultPrng = .init(testing.random_seed);
    const random = prng.random();

    var w1: f32 = random.float(f32);
    var w2: f32 = random.float(f32);
    var b: f32 = random.float(f32);

    const h: f32 = 1e-3;
    const rate: f32 = 1e-1;
    for (0..10000) |_| {
        const c = cost(w1, w2, b);

        const gradient_w1 = (cost(w1 + h, w2, b) - c) / h;
        const gradient_w2 = (cost(w1, w2 + h, b) - c) / h;
        const gradient_b = (cost(w1, w2, b + h) - c) / h;

        w1 -= rate * gradient_w1;
        w2 -= rate * gradient_w2;
        b -= rate * gradient_b;
    }

    for (train_data) |x| {
        std.debug.print(
            "{d} | {d} = {d} => {d}\n",
            .{
                x[0],
                x[1],
                x[2],
                sigmoid(x[0] * w1 + x[1] * w2 + b),
            },
        );
    }
}

fn cost(w1: f32, w2: f32, b: f32) f32 {
    var mean_squared_error: f32 = 0;
    for (train_data) |data| {
        const actual = data[2];
        const x1 = data[0];
        const x2 = data[1];

        const y = sigmoid(x1 * w1 + x2 * w2 + b);
        const err = y - actual;

        mean_squared_error += err * err;
    }
    mean_squared_error /= train_data.len;
    return mean_squared_error;
}

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + std.math.exp(-x));
}

test "sigmoid" {
    var x: f32 = -10;
    while (x < 10) : (x += 1) {
        try testing.expect(sigmoid(x) > 0);
        try testing.expect(sigmoid(x) < 1);
    }
}
