struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    // fullscreen triangle
    let x = f32(i32(in_vertex_index & 1u) * 4 - 1);
    let y = f32(i32(in_vertex_index > 1u) * 4 - 1);
    return VertexOutput(vec4<f32>(x, y, 0.0, 1.0), vec2<f32>(x, y) * 0.5 + 0.5);
}

@group(0)
@binding(1)
var tex: texture_2d<f32>;
@group(0)
@binding(2)
var sam: sampler;

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    let sampleLoc = vec2<i32>(vertex.tex_coords) * vec2<i32>(100, 100);
    return vec4<f32>(textureLoad(tex, sampleLoc, 0).r + 0.5, 0.0, 0.0, 1.0);
}
