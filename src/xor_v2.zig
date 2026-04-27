const std = @import("std");
const neural_network = @import("neural_network");
const assert = std.debug.assert;
const testing = std.testing;

const Io = std.Io;
const Allocator = std.mem.Allocator;
const Matrix = neural_network.matrix.Matrix(f32);
const NeuralNetwork = neural_network.NeuralNetwork;

var xor = [_]f32{
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stderr(), io, &stdout_buffer);
    const stdout_writer = &stdout_file_writer.interface;

    var prng: std.Random.DefaultPrng = .init(67);
    const random = prng.random();

    const training_data: Matrix = .init_from_slice(&xor, .init(4, 3));
    var training_input: Matrix = try training_data.copy_submatrix(arena, .{
        .inital = 0,
        .stride = 1,
        .column = 2,
    });
    var training_target: Matrix = try training_data.copy_submatrix(arena, .{
        .inital = 2,
        .stride = 1,
        .column = 1,
    });

    var nn: NeuralNetwork = try .init(arena, .{ .architecture = &.{ 2, 3, 1 } });
    nn.fill_rand(random);

    const epoch = 100_000;
    for (0..epoch) |_| {
        for (0..training_data.shape.row) |row| {
            nn.forward(try training_input.copy_row(arena, row));
            nn.backward(try training_target.copy_row(arena, row));
            nn.learn(0.01);
        }
    }

    for (0..training_data.shape.row) |row| {
        nn.forward(try training_input.copy_row(arena, row));

        try stdout_writer.print("{d} ^ {d} = {d}\n", .{
            training_input.at(row, 0),
            training_input.at(row, 1),
            nn.activations[nn.activations.len - 1].at(0, 0),
        });
    }
    try stdout_writer.flush();
}
