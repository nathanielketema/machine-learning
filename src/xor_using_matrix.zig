//! Naive xor implementation
const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;

const Io = std.Io;
const Allocator = std.mem.Allocator;

const matrix = @import("matrix");
const Matrix = matrix.Matrix(f32);

const eps: f32 = 1e-1;
const learning_rate: f32 = 1e-1;

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
const row: Matrix.Row = .to_enum(4);
const col: Matrix.Col = .to_enum(3);

const XorModel = struct {
    x: Matrix,

    w1: Matrix,
    b1: Matrix,
    a1: Matrix,

    w2: Matrix,
    b2: Matrix,
    a2: Matrix,
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();

    var prng: std.Random.DefaultPrng = .init(67);
    const random = prng.random();

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const stdout_writer = &stdout_file_writer.interface;

    var xor: XorModel = .{
        .x = try .init(arena, .to_enum(1), .to_enum(2)),

        .w1 = try .init(arena, .to_enum(2), .to_enum(2)),
        .b1 = try .init(arena, .to_enum(1), .to_enum(2)),
        .a1 = try .init(arena, .to_enum(1), .to_enum(2)),

        .w2 = try .init(arena, .to_enum(2), .to_enum(1)),
        .b2 = try .init(arena, .to_enum(1), .to_enum(1)),
        .a2 = try .init(arena, .to_enum(1), .to_enum(1)),
    };

    xor.w1.fill_random(random);
    xor.b1.fill_random(random);
    xor.w2.fill_random(random);
    xor.b2.fill_random(random);

    const train_data: Matrix = .create(&xor_gate, row, col);

    const train_input = try train_data.slice(arena, 0, .to_enum(2));
    const train_output = try train_data.slice(arena, 2, .to_enum(2));

    try stdout_writer.print("cost = {d}\n", .{cost(arena, &xor, train_input, train_output)});
    const predicted = xor.a2.get(0, 0);
    for (0..train_input.row.value()) |r| {
        try stdout_writer.print(
            "{d} ^ {d} = {d}\n",
            .{
                train_input.get(r, 0),
                train_input.get(r, 1),
                predicted,
            },
        );
    }

    try stdout_writer.flush();
}

fn cost(gpa: Allocator, xor: *XorModel, train_input: Matrix, train_output: Matrix) f32 {
    assert(train_input.row.value() == train_output.row.value());
    assert(train_output.col.value() == xor.a2.col.value());

    const input_row_count = train_input.row.value();
    const output_col_count = train_output.col.value();

    var loss: f32 = 0;
    for (0..input_row_count) |ro| {
        const r: u16 = @intCast(ro);
        xor.x = train_input.row_as_matrix(gpa, .to_enum(r)) catch unreachable;
        const actual = train_output.row_as_matrix(gpa, .to_enum(r)) catch unreachable;
        forward(xor);

        for (0..output_col_count) |c| {
            const diff = actual.get(0, c) - xor.a2.get(0, c);
            loss += diff * diff;
        }
    }
    return loss / @as(f32, input_row_count);
}

fn forward(xor: *XorModel) void {
    // a1 = x*w1 + b1
    matrix.dot(f32, &xor.a1, xor.x, xor.w1);
    matrix.add(f32, &xor.a1, xor.a1, xor.b1);
    apply_sigmoid(&xor.a1);

    // a2 = a1*w2 + b2
    matrix.dot(f32, &xor.a2, xor.a1, xor.w2);
    matrix.add(f32, &xor.a2, xor.a2, xor.b2);
    apply_sigmoid(&xor.a2);
}

fn sigmoid(x: f32) f32 {
    return 1 / (1 + std.math.exp(-x));
}

fn apply_sigmoid(m: *Matrix) void {
    for (m.data) |*data| {
        data.* = sigmoid(data.*);
    }
}
