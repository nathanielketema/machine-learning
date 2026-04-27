const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    var nn_mod = b.dependency("neural_network", .{
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "machine_learning",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/xor_v2.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "neural_network", .module = nn_mod.module("neural_network") },
            },
        }),
    });
    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_step.dependOn(b.getInstallStep());
}
