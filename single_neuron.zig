//! A single neural network
//! - checkout [YouTube](https://www.youtube.com/watch?v=PGSba51aRYU&list=PLpM-Dvs8t0VZPZKggcql-MmjaBdZKeDMw)
//!

const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;

// # Neural Networks
//
// The simple formula (it's similar to the line formula):
// - f(x)/y = x * weight + bias;
//
// In general:
// - f(x)/y = x1*w1 + x2*w2 + .... + b
//
// The model is trying to learn the weight needed to get the desired output
//
// Steps:
// 1. pick a random weight
// 2. find y = sum(x * weight) + bias (forwarding)
//    - depending on your output, apply:
//    - y = activation_function(y) -> turns the result into a probability (0-1)
// 4. calculate the cost function (diff between actual and expected)
// 5. back propagation  (finite_difference)
// 6. adjustment

const training_data = [_][2]f32{ // {input(x), output(y)}
    .{ 0, 0 },
    .{ 1, 2 },
    .{ 2, 4 },
    .{ 3, 6 },
    .{ 4, 8 },
};

pub fn main(_: std.process.Init) !void {
    var prng: std.Random.DefaultPrng = .init(testing.random_seed);
    const random = prng.random();

    var weight: f32 = random.float(f32) * 10;
    var bias: f32 = random.float(f32);

    std.debug.print("Inital: weight = {d}, bias = {d}, cost(weight, bias) = {d}\n", .{
        weight,
        bias,
        cost(weight, bias),
    });

    // gradient_weight = finite_difference/derivative
    // we use rate(also called the learning rate) to speed up the weight change
    const h: f32 = 1e-3;
    const rate: f32 = 1e-3;
    for (0..500) |_| {
        // To not calculate it twice
        const c = cost(weight, bias);
        const gradient_weight = (cost(weight + h, bias) - c) / h;
        const gradient_bias = (cost(weight, bias + h) - c) / h;

        // Update
        weight -= rate * gradient_weight;
        bias -= rate * gradient_bias;
    }

    std.debug.print("---\n", .{});
    std.debug.print("Final: weight = {d}, bias = {d}, cost(weight, bias) = {d}\n", .{
        weight,
        bias,
        cost(weight, bias),
    });

    std.debug.print("\nx | y = Model\n", .{});
    std.debug.print("-------------\n", .{});
    for (0..training_data.len) |i| {
        std.debug.print(
            "{d} | {d} = {d}\n",
            .{
                training_data[i][0],
                training_data[i][1],
                training_data[i][0] * weight + bias,
            },
        );
    }
}

// The goal is for:
// - cost(weight) => 0
//
// By squaring the error we ensure our cost function is curvy (parabolla) which helps with
// calculating the finite_difference later
fn cost(weight: f32, bias: f32) f32 {
    var mean_squared_error: f32 = 0;
    for (training_data) |data| {
        const actual = data[1];
        const x = data[0];

        const y = x * weight + bias;
        const err = y - actual;

        mean_squared_error += err * err;
    }
    mean_squared_error /= training_data.len;

    return mean_squared_error;
}
