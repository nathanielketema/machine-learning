//! A single neural network
//! - checkout [YouTube](https://www.youtube.com/watch?v=PGSba51aRYU&list=PLpM-Dvs8t0VZPZKggcql-MmjaBdZKeDMw)

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const testing = std.testing;


const train = [_][2]f16{
    // {input(x), output(y)}
    .{ 0, 0 },
    .{ 1, 2 },
    .{ 2, 4 },
    .{ 3, 6 },
    .{ 4, 8 },
};

pub fn main() !void {
    var prng: std.Random.DefaultPrng = .init(69);
    const random = prng.random();

    // The model is trying to prdict w
    // y = x * w;
    
    var w: f32 = random.float(f32) * 10; // 7
    std.debug.print("w: {d} \n", .{w});
    std.debug.print("{d} \n", .{cost(w)});

    const eps: f16 = 1e-3;
    const rate: f16 = 1e-3;
    for (0..500) |_| {
        const delta_cost = (cost(w + eps) - cost(w)) / eps;
        w -= rate * delta_cost;
        std.debug.print("cost = {d}, w = {d}\n", .{cost(w), w});
    }
    std.debug.print("--------------------- \n", .{});
    std.debug.print("w: {d} \n", .{w});
}

fn cost(w: f32) f32 {
    var result: f32 = 0;
    for (train) |data| {
        const x = data[0];
        const y = x * w;
        const distance = y - data[1];
        result += distance * distance;
    }
    result /= train.len;
    return result;
}

test "foo" {}
