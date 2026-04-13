//! Naive xor implementation
//!
//! Possible names:
//! - pred
//! - goal
//! - true
//! - bias
//! - diff
//! - wght
//! - loss
//! - cost
//! - lr
const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;

const Matrix = @import("matrix").Matrix(f32);
const Io = std.Io;
const Allocator = std.mem.Allocator;

const TrainOptions = struct {
    eps: f32 = 1e-1,
    learning_rate: f32 = 1e-1,
    epochs: usize = 1000,
};

var or_gate = [_]f32{
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 1,
};

var and_gate = [_]f32{
    0, 0, 0,
    0, 1, 0,
    1, 0, 0,
    1, 1, 1,
};

var nand_gate = [_]f32{
    0, 0, 1,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

var xor_gate = [_]f32{
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};
const dataset: Matrix = .init_from_slice(&xor_gate, .init(4, 3));

const TrainingData = struct {
    input: Matrix,
    output: Matrix,

    pub fn init(arena: Allocator, input_data: Matrix) !TrainingData {
        return .{
            .input = try input_data.copy_submatrix(arena, .{
                .start_col = 0,
                .stride = 2,
            }),
            .output = try input_data.copy_submatrix(arena, .{
                .start_col = 2,
                .stride = 2,
            }),
        };
    }
};

const XorModel = struct {
    x: Matrix,

    w1: Matrix,
    b1: Matrix,
    a1: Matrix,

    w2: Matrix,
    b2: Matrix,
    a2: Matrix,

    fn init(arena: Allocator) !XorModel {
        return .{
            .x = try .init(arena, .init(1, 2)),
            .w1 = try .init(arena, .init(2, 2)),
            .b1 = try .init(arena, .init(1, 2)),
            .a1 = try .init(arena, .init(1, 2)),
            .w2 = try .init(arena, .init(2, 1)),
            .b2 = try .init(arena, .init(1, 1)),
            .a2 = try .init(arena, .init(1, 1)),
        };
    }

    pub fn fill_random(xor: *XorModel, random: std.Random) void {
        xor.w1.fill_random(random);
        xor.b1.fill_random(random);
        xor.w2.fill_random(random);
        xor.b2.fill_random(random);
    }

    pub fn forward(xor: *XorModel) void {
        // a1 = x*w1 + b1
        xor.a1.mul(xor.x, xor.w1);
        xor.a1.add(xor.a1, xor.b1);
        apply_sigmoid(&xor.a1);

        xor.a2.mul(xor.a1, xor.w2);
        xor.a2.add(xor.a2, xor.b2);
        apply_sigmoid(&xor.a2);
    }
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();
    var prng: std.Random.DefaultPrng = .init(67);
    const random = prng.random();
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const stdout_writer = &stdout_file_writer.interface;

    var xor: XorModel = try .init(arena);
    xor.fill_random(random);

    const training_data: TrainingData = try .init(arena, dataset);
    try stdout_writer.print("Before cost = {d}\n", .{cost(arena, &xor, training_data)});
    try stdout_writer.print("---\n", .{});

    const predicted = xor.a2.at(0, 0);
    for (0..training_data.input.shape.row) |r| {
        try stdout_writer.print(
            "{d} ^ {d} = {d}\n",
            .{
                training_data.input.at(r, 0),
                training_data.input.at(r, 1),
                predicted,
            },
        );
    }

    try stdout_writer.flush();
}

fn cost(gpa: Allocator, xor: *XorModel, sample: TrainingData) f32 {
    assert(sample.input.shape.row == sample.output.shape.row);
    assert(sample.output.shape.col == xor.a2.shape.col);

    const input_row_count = sample.input.shape.row;
    const output_col_count = sample.output.shape.col;

    var loss: f32 = 0;
    for (0..input_row_count) |r| {
        xor.x = sample.input.copy_row(gpa, r) catch unreachable;
        const actual = sample.output.copy_row(gpa, r) catch unreachable;
        xor.forward();

        for (0..output_col_count) |c| {
            const diff = actual.at(0, c) - xor.a2.at(0, c);
            loss += diff * diff;
        }
    }
    return loss / @as(f32, input_row_count);
}

fn sigmoid(x: f32) f32 {
    return 1 / (1 + std.math.exp(-x));
}

fn apply_sigmoid(m: *Matrix) void {
    for (m.data) |*data| {
        data.* = sigmoid(data.*);
    }
}
