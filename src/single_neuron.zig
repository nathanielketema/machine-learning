//! A single neural network
//! - checkout [YouTube](https://www.youtube.com/watch?v=PGSba51aRYU&list=PLpM-Dvs8t0VZPZKggcql-MmjaBdZKeDMw)
//!

const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;

// # Neural Networks
//
// The simple formula (it's similar to the line formula):
// - f(input)/pred = input * wght + bias;
//
// In general:
// - f(input)/pred = x1*w1 + x2*w2 + .... + b
//
// The model is trying to learn the wght needed to get the desired output
//
// Steps:
// 1. pick a random wght
// 2. find pred = sum(input * wght) + bias (forwarding)
//    - depending on your output, apply:
//    - pred = activation_function(pred) -> turns the result into a probability (0-1)
// 4. calculate the cost function (diff between goal and expected)
// 5. back propagation  (finite_difference)
// 6. adjustment

const training_data = [_][2]f32{ // {input(input), output(pred)}
    .{ 0, 0 },
    .{ 1, 2 },
    .{ 2, 4 },
    .{ 3, 6 },
    .{ 4, 8 },
};

pub fn main(_: std.process.Init) !void {
    var prng: std.Random.DefaultPrng = .init(testing.random_seed);
    const random = prng.random();

    var wght: f32 = random.float(f32) * 10;
    var bias: f32 = random.float(f32);

    std.debug.print("Inital: wght = {d}, bias = {d}, cost(wght, bias) = {d}\n", .{
        wght,
        bias,
        cost(wght, bias),
    });

    // wght_grad = finite_difference/derivative
    // we use alpha(also called the learning alpha) to speed up the wght change
    const h: f32 = 1e-3;
    const alpha: f32 = 1e-3;
    const epoch: usize = 500;
    for (0..epoch) |_| {
        // To not calculate it twice
        const loss = cost(wght, bias);
        const wght_grad = (cost(wght + h, bias) - loss) / h;
        const bias_grad = (cost(wght, bias + h) - loss) / h;

        // Update
        wght -= alpha * wght_grad;
        bias -= alpha * bias_grad;
    }

    std.debug.print("---\n", .{});
    std.debug.print("Final: wght = {d}, bias = {d}, cost(wght, bias) = {d}\n", .{
        wght,
        bias,
        cost(wght, bias),
    });

    std.debug.print("\nx | pred = Model\n", .{});
    std.debug.print("-------------\n", .{});
    for (training_data) |data| {
        const pred = data[0] * wght + bias;
        std.debug.print(
            "{d} | {d} = {d}\n",
            .{
                data[0],
                data[1],
                pred,
            },
        );
    }
}

// The goal is for:
// - cost(wght) => 0
//
// By squaring the error we ensure our cost function is curvy (parabolla) which helps with
// calculating the finite_difference later
fn cost(wght: f32, bias: f32) f32 {
    var mean_squared_error: f32 = 0;
    for (training_data) |data| {
        const pred = data[0] * wght + bias;
        const goal = data[1];

        const diff = pred - goal;

        mean_squared_error += diff * diff;
    }
    return mean_squared_error / training_data.len;
}
