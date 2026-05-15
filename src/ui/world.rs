use std::sync::Arc;

use anyhow::{Context as _, Result};
use bytemuck::{Pod, Zeroable};
use eframe::egui::{self, Color32, Rect, Sense, Stroke};
use eframe::egui_wgpu;
use eframe::wgpu;
use eframe::wgpu::util::DeviceExt as _;

use crate::settings::UiConfig;
use crate::sim::RenderSnapshotReadback;
use crate::ui::render;
use crate::ui::types::{CameraState, SelectedAgentData};

const GRID_DIVISIONS: u32 = 12;
const QUAD_VERTICES: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
const QUAD_INDICES: [u16; 6] = [0, 1, 2, 0, 2, 3];
const TRIANGLE_INDICES: [u16; 3] = [0, 1, 2];
const QUAD_VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 1] =
    [wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 }];
const FOOD_INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 3] = [
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 1 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 8, shader_location: 2 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 12, shader_location: 3 },
];
const AGENT_INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 1 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 8, shader_location: 2 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 16, shader_location: 3 },
    wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 20, shader_location: 4 },
];

#[derive(Clone)]
pub struct WorldFrame {
    pub snapshot: RenderSnapshotReadback,
    food_instances: Vec<FoodInstance>,
    prey_instances: Vec<AgentInstance>,
    predator_instances: Vec<AgentInstance>,
}

impl WorldFrame {
    pub const fn empty() -> Self {
        Self {
            snapshot: RenderSnapshotReadback::empty(),
            food_instances: Vec::new(),
            prey_instances: Vec::new(),
            predator_instances: Vec::new(),
        }
    }

    pub fn from_snapshot(snapshot: RenderSnapshotReadback, ui_config: &UiConfig) -> Self {
        let mut frame = Self::empty();
        frame.snapshot = snapshot;
        frame.rebuild_instances(ui_config);
        frame
    }

    pub fn rebuild_instances(&mut self, ui_config: &UiConfig) {
        let food_color = rgba_with_alpha(ui_config.food_color, alpha_to_unit(ui_config.food_alpha));
        let prey_color = rgba_with_alpha(ui_config.prey_color, 1.0);
        let predator_color = rgba_with_alpha(ui_config.predator_color, 1.0);

        self.food_instances.clear();
        self.food_instances.extend(self.snapshot.food.iter().map(|entry| FoodInstance {
            pos: [entry.pos_x, entry.pos_y],
            radius: ui_config.food_size * 0.5,
            color: food_color,
        }));

        self.prey_instances.clear();
        self.prey_instances.extend(self.snapshot.prey.iter().map(|entry| {
            AgentInstance::from_readback(
                entry.pos_x,
                entry.pos_y,
                entry.dir_x,
                entry.dir_y,
                ui_config.prey_size,
                prey_color,
            )
        }));

        self.predator_instances.clear();
        self.predator_instances.extend(self.snapshot.predators.iter().map(|entry| {
            AgentInstance::from_readback(
                entry.pos_x,
                entry.pos_y,
                entry.dir_x,
                entry.dir_y,
                ui_config.predator_size,
                predator_color,
            )
        }));
    }

    pub const fn active_food_count(&self) -> u32 {
        self.snapshot.header.returned_food
    }
}

pub fn install_renderer_resources(render_state: &egui_wgpu::RenderState, ui_config: &UiConfig) -> Result<()> {
    let mut renderer = render_state.renderer.write();
    if renderer.callback_resources.get::<WorldRenderResources>().is_some() {
        return Ok(());
    }

    let resources = WorldRenderResources::new(&render_state.device, render_state.target_format, ui_config)
        .context("failed to initialize world renderer resources")?;
    renderer.callback_resources.insert(resources);
    Ok(())
}

pub fn refresh_renderer_resources(render_state: &egui_wgpu::RenderState, ui_config: &UiConfig) -> Result<()> {
    let mut renderer = render_state.renderer.write();
    let resources = WorldRenderResources::new(&render_state.device, render_state.target_format, ui_config)
        .context("failed to refresh world renderer resources")?;
    renderer.callback_resources.insert(resources);
    Ok(())
}

pub fn allocate_world_rect(ui: &mut egui::Ui) -> (Rect, egui::Response) {
    ui.allocate_exact_size(ui.available_size().max(egui::vec2(1.0, 1.0)), Sense::click_and_drag())
}

pub fn paint_world(
    ui: &mut egui::Ui,
    rect: Rect,
    frame: Arc<WorldFrame>,
    ui_config: &UiConfig,
    camera: CameraState,
    world_size: f32,
    selected: Option<&SelectedAgentData>,
    vision_range: f32,
) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, rgb(ui_config.bg_color));
    paint_grid(&painter, rect, ui_config, camera, world_size);
    painter.add(egui_wgpu::Callback::new_paint_callback(
        rect,
        WorldPaintCallback {
            frame,
            camera,
            world_size,
            viewport_size: [rect.width().max(1.0), rect.height().max(1.0)],
        },
    ));
    if let Some(selected) = selected {
        paint_selected_overlay(&painter, rect, selected, ui_config, camera, world_size, vision_range);
    }
}

#[derive(Clone)]
struct WorldPaintCallback {
    frame: Arc<WorldFrame>,
    camera: CameraState,
    world_size: f32,
    viewport_size: [f32; 2],
}

impl egui_wgpu::CallbackTrait for WorldPaintCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(resources) = callback_resources.get_mut::<WorldRenderResources>() else {
            return Vec::new();
        };
        resources.prepare(device, queue, &self.frame, self.camera, self.world_size, self.viewport_size);
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        if let Some(resources) = callback_resources.get::<WorldRenderResources>() {
            resources.paint(render_pass, &self.frame);
        }
    }
}

struct WorldRenderResources {
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    disc_pipeline: wgpu::RenderPipeline,
    triangle_pipeline: wgpu::RenderPipeline,
    quad_vertex_buffer: wgpu::Buffer,
    quad_index_buffer: wgpu::Buffer,
    triangle_vertex_buffer: wgpu::Buffer,
    triangle_index_buffer: wgpu::Buffer,
    food_buffer: ResizableBuffer,
    prey_buffer: ResizableBuffer,
    predator_buffer: ResizableBuffer,
}

impl WorldRenderResources {
    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat, ui_config: &UiConfig) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("moonai_world_shader"),
            source: wgpu::ShaderSource::Wgsl(WORLD_SHADER.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("moonai_world_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        wgpu::BufferSize::new(u64::try_from(std::mem::size_of::<SceneUniform>())?)
                            .context("scene uniform minimum binding size overflowed")?,
                    ),
                },
                count: None,
            }],
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("moonai_world_uniform_buffer"),
            size: u64::try_from(std::mem::size_of::<SceneUniform>()).context("scene uniform size overflowed")?,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("moonai_world_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("moonai_world_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let quad_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("moonai_world_quad_vertices"),
            contents: bytemuck::cast_slice(&QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let quad_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("moonai_world_quad_indices"),
            contents: bytemuck::cast_slice(&QUAD_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let triangle_vertices = triangle_vertices(ui_config);
        let triangle_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("moonai_world_triangle_vertices"),
            contents: bytemuck::cast_slice(&triangle_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let triangle_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("moonai_world_triangle_indices"),
            contents: bytemuck::cast_slice(&TRIANGLE_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let disc_pipeline = create_pipeline(
            device,
            &pipeline_layout,
            &shader,
            target_format,
            &[quad_vertex_layout(), food_instance_layout()],
            "food_vertex",
            "circle_fragment",
            "moonai_world_disc_pipeline",
        );
        let triangle_pipeline = create_pipeline(
            device,
            &pipeline_layout,
            &shader,
            target_format,
            &[triangle_vertex_layout(), agent_instance_layout()],
            "agent_vertex",
            "flat_fragment",
            "moonai_world_triangle_pipeline",
        );

        Ok(Self {
            uniform_buffer,
            bind_group,
            disc_pipeline,
            triangle_pipeline,
            quad_vertex_buffer,
            quad_index_buffer,
            triangle_vertex_buffer,
            triangle_index_buffer,
            food_buffer: ResizableBuffer::new::<FoodInstance>(device, "moonai_world_food_instances"),
            prey_buffer: ResizableBuffer::new::<AgentInstance>(device, "moonai_world_prey_instances"),
            predator_buffer: ResizableBuffer::new::<AgentInstance>(device, "moonai_world_predator_instances"),
        })
    }

    fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &WorldFrame,
        camera: CameraState,
        world_size: f32,
        viewport_size: [f32; 2],
    ) {
        let uniform = SceneUniform {
            camera_center: [camera.center_x, camera.center_y],
            viewport_size,
            world_size,
            zoom: camera.zoom,
            padding: [0.0, 0.0],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        self.food_buffer.write(device, queue, &frame.food_instances);
        self.prey_buffer.write(device, queue, &frame.prey_instances);
        self.predator_buffer.write(device, queue, &frame.predator_instances);
    }

    fn paint(&self, render_pass: &mut wgpu::RenderPass<'static>, frame: &WorldFrame) {
        render_pass.set_bind_group(0, &self.bind_group, &[]);

        if !frame.food_instances.is_empty() {
            render_pass.set_pipeline(&self.disc_pipeline);
            render_pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.food_buffer.buffer.slice(..));
            render_pass.set_index_buffer(self.quad_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..QUAD_INDICES.len() as u32, 0, 0..frame.food_instances.len() as u32);
        }

        if !frame.prey_instances.is_empty() {
            render_pass.set_pipeline(&self.triangle_pipeline);
            render_pass.set_vertex_buffer(0, self.triangle_vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.prey_buffer.buffer.slice(..));
            render_pass.set_index_buffer(self.triangle_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..TRIANGLE_INDICES.len() as u32, 0, 0..frame.prey_instances.len() as u32);
        }

        if !frame.predator_instances.is_empty() {
            render_pass.set_pipeline(&self.triangle_pipeline);
            render_pass.set_vertex_buffer(0, self.triangle_vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.predator_buffer.buffer.slice(..));
            render_pass.set_index_buffer(self.triangle_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..TRIANGLE_INDICES.len() as u32, 0, 0..frame.predator_instances.len() as u32);
        }
    }
}

struct ResizableBuffer {
    buffer: wgpu::Buffer,
    capacity: usize,
    label: &'static str,
    usage: wgpu::BufferUsages,
}

impl ResizableBuffer {
    fn new<T: Pod>(device: &wgpu::Device, label: &'static str) -> Self {
        let usage = wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST;
        Self { buffer: create_vertex_buffer::<T>(device, label, 1, usage), capacity: 1, label, usage }
    }

    fn write<T: Pod>(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, items: &[T]) {
        let required = items.len().max(1).next_power_of_two();
        if required > self.capacity {
            self.buffer = create_vertex_buffer::<T>(device, self.label, required, self.usage);
            self.capacity = required;
        }
        if !items.is_empty() {
            queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(items));
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneUniform {
    camera_center: [f32; 2],
    viewport_size: [f32; 2],
    world_size: f32,
    zoom: f32,
    padding: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FoodInstance {
    pos: [f32; 2],
    radius: f32,
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AgentInstance {
    pos: [f32; 2],
    dir: [f32; 2],
    radius: f32,
    color: [f32; 4],
}

impl AgentInstance {
    fn from_readback(pos_x: f32, pos_y: f32, dir_x: f32, dir_y: f32, radius: f32, color: [f32; 4]) -> Self {
        let magnitude = (dir_x * dir_x + dir_y * dir_y).sqrt();
        let dir = if magnitude <= f32::EPSILON { [1.0, 0.0] } else { [dir_x / magnitude, dir_y / magnitude] };
        Self { pos: [pos_x, pos_y], dir, radius, color }
    }
}

fn paint_grid(painter: &egui::Painter, rect: Rect, ui_config: &UiConfig, camera: CameraState, world_size: f32) {
    let grid_stroke = Stroke::new(1.0, rgb(ui_config.grid_color));
    let border_stroke = Stroke::new(1.0, rgb(ui_config.border_color));

    for index in 1..GRID_DIVISIONS {
        let world = world_size * (index as f32 / GRID_DIVISIONS as f32);
        let vertical_top = render::world_to_screen(rect, camera, world_size, world, world_size);
        let vertical_bottom = render::world_to_screen(rect, camera, world_size, world, 0.0);
        painter.line_segment([vertical_top, vertical_bottom], grid_stroke);

        let horizontal_left = render::world_to_screen(rect, camera, world_size, 0.0, world);
        let horizontal_right = render::world_to_screen(rect, camera, world_size, world_size, world);
        painter.line_segment([horizontal_left, horizontal_right], grid_stroke);
    }

    let top_left = render::world_to_screen(rect, camera, world_size, 0.0, world_size);
    let top_right = render::world_to_screen(rect, camera, world_size, world_size, world_size);
    let bottom_right = render::world_to_screen(rect, camera, world_size, world_size, 0.0);
    let bottom_left = render::world_to_screen(rect, camera, world_size, 0.0, 0.0);
    painter.line_segment([top_left, top_right], border_stroke);
    painter.line_segment([top_right, bottom_right], border_stroke);
    painter.line_segment([bottom_right, bottom_left], border_stroke);
    painter.line_segment([bottom_left, top_left], border_stroke);
}

fn paint_selected_overlay(
    painter: &egui::Painter,
    rect: Rect,
    selected: &SelectedAgentData,
    ui_config: &UiConfig,
    camera: CameraState,
    world_size: f32,
    vision_range: f32,
) {
    let center = render::world_to_screen(rect, camera, world_size, selected.agent.pos_x, selected.agent.pos_y);
    let selection_radius = scaled_radius(
        rect,
        camera,
        world_size,
        agent_selection_radius_world(ui_config.predator_size.max(ui_config.prey_size), ui_config),
    ) + ui_config.selected_outline_thickness
        + 2.0;
    painter.circle_stroke(
        center,
        selection_radius,
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 220)),
    );

    let vision_radius = scaled_radius(rect, camera, world_size, vision_range).max(2.0);
    painter.circle_stroke(
        center,
        vision_radius,
        Stroke::new(
            1.0,
            Color32::from_rgba_unmultiplied(
                unit_to_u8(ui_config.vision_outline_color[0]),
                unit_to_u8(ui_config.vision_outline_color[1]),
                unit_to_u8(ui_config.vision_outline_color[2]),
                alpha_to_u8(ui_config.vision_outline_alpha),
            ),
        ),
    );

    for (world_x, world_y, group) in sensor_lines(selected, vision_range) {
        let target = render::world_to_screen(rect, camera, world_size, world_x, world_y);
        let alpha =
            if group == 2 { alpha_to_u8(ui_config.food_sensor_alpha) } else { alpha_to_u8(ui_config.sensor_alpha) };
        let color = match group {
            0 => rgba_color32(ui_config.predator_color, alpha),
            1 => rgba_color32(ui_config.prey_color, alpha),
            _ => rgba_color32(ui_config.food_color, alpha),
        };
        painter.line_segment([center, target], Stroke::new(1.0, color));
    }
}

fn sensor_lines(selected: &SelectedAgentData, vision_range: f32) -> Vec<(f32, f32, u8)> {
    let mut lines = Vec::with_capacity(15);
    for group in 0..3 {
        let start = group * 10;
        for pair in 0..5 {
            let dx = selected.sensors.inputs[start + (pair * 2)];
            let dy = selected.sensors.inputs[start + (pair * 2) + 1];
            if dx == 0.0 && dy == 0.0 {
                continue;
            }
            lines.push((
                selected.agent.pos_x + (dx * vision_range),
                selected.agent.pos_y + (dy * vision_range),
                group as u8,
            ));
        }
    }
    lines
}

fn scaled_radius(rect: Rect, camera: CameraState, world_size: f32, radius_world: f32) -> f32 {
    let shortest = rect.width().min(rect.height()).max(1.0);
    ((shortest / world_size.max(1.0)) * camera.zoom * radius_world).max(0.0)
}

fn triangle_vertices(ui_config: &UiConfig) -> [[f32; 2]; 3] {
    let tip = ui_config.triangle_tip_factor.max(0.0);
    let base = ui_config.triangle_base_factor.max(0.0);
    let width = ui_config.triangle_width_factor.max(0.0);
    let max_extent = (tip + base).max(width * 2.0).max(1.0);
    let scale = 1.0 / max_extent;
    [[tip * scale, 0.0], [-base * scale, width * scale], [-base * scale, -width * scale]]
}

fn agent_selection_radius_world(size: f32, ui_config: &UiConfig) -> f32 {
    triangle_vertices(ui_config).iter().map(|vertex| vertex[0].hypot(vertex[1]) * size).fold(0.0_f32, f32::max)
}

fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    target_format: wgpu::TextureFormat,
    buffers: &[wgpu::VertexBufferLayout<'_>],
    vertex_entry: &str,
    fragment_entry: &str,
    label: &'static str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(vertex_entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers,
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fragment_entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

const fn quad_vertex_layout<'a>() -> wgpu::VertexBufferLayout<'a> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<[f32; 2]>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &QUAD_VERTEX_ATTRIBUTES,
    }
}

const fn triangle_vertex_layout<'a>() -> wgpu::VertexBufferLayout<'a> {
    quad_vertex_layout()
}

const fn food_instance_layout<'a>() -> wgpu::VertexBufferLayout<'a> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<FoodInstance>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &FOOD_INSTANCE_ATTRIBUTES,
    }
}

const fn agent_instance_layout<'a>() -> wgpu::VertexBufferLayout<'a> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<AgentInstance>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &AGENT_INSTANCE_ATTRIBUTES,
    }
}

fn create_vertex_buffer<T: Pod>(
    device: &wgpu::Device,
    label: &'static str,
    capacity: usize,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let element_size = std::mem::size_of::<T>() as u64;
    let capacity = capacity.max(1) as u64;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: element_size.saturating_mul(capacity),
        usage,
        mapped_at_creation: false,
    })
}

fn rgb(color: [f32; 3]) -> Color32 {
    Color32::from_rgb(unit_to_u8(color[0]), unit_to_u8(color[1]), unit_to_u8(color[2]))
}

fn rgba_color32(color: [f32; 3], alpha: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(unit_to_u8(color[0]), unit_to_u8(color[1]), unit_to_u8(color[2]), alpha)
}

const fn rgba_with_alpha(color: [f32; 3], alpha: f32) -> [f32; 4] {
    [color[0], color[1], color[2], alpha.clamp(0.0, 1.0)]
}

fn alpha_to_unit(alpha: f32) -> f32 {
    (alpha / 255.0).clamp(0.0, 1.0)
}

const fn alpha_to_u8(alpha: f32) -> u8 {
    alpha.clamp(0.0, 255.0) as u8
}

fn unit_to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

const WORLD_SHADER: &str = r#"
struct SceneUniform {
    camera_center: vec2<f32>,
    viewport_size: vec2<f32>,
    world_size: f32,
    zoom: f32,
    padding: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> scene: SceneUniform;

struct FoodVertexInput {
    @location(0) local: vec2<f32>,
    @location(1) pos: vec2<f32>,
    @location(2) radius: f32,
    @location(3) color: vec4<f32>,
};

struct AgentVertexInput {
    @location(0) local: vec2<f32>,
    @location(1) pos: vec2<f32>,
    @location(2) dir: vec2<f32>,
    @location(3) radius: f32,
    @location(4) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) color: vec4<f32>,
};

fn pixels_per_world() -> f32 {
    let shortest = max(min(scene.viewport_size.x, scene.viewport_size.y), 1.0);
    return (shortest / max(scene.world_size, 1.0)) * scene.zoom;
}

fn world_to_ndc(world: vec2<f32>) -> vec2<f32> {
    let scale = pixels_per_world();
    // Match the CPU world_to_screen transform so overlays and GPU geometry stay aligned.
    let centered = vec2<f32>(world.x - scene.camera_center.x, world.y - scene.camera_center.y);
    return vec2<f32>(
        (2.0 * centered.x * scale) / scene.viewport_size.x,
        (2.0 * centered.y * scale) / scene.viewport_size.y,
    );
}

fn world_delta_to_ndc(delta: vec2<f32>) -> vec2<f32> {
    let scale = pixels_per_world();
    return vec2<f32>(
        (2.0 * delta.x * scale) / scene.viewport_size.x,
        (2.0 * delta.y * scale) / scene.viewport_size.y,
    );
}

@vertex
fn food_vertex(input: FoodVertexInput) -> VertexOutput {
    let world = input.pos + (input.local * input.radius);
    var out: VertexOutput;
    out.position = vec4<f32>(world_to_ndc(world), 0.0, 1.0);
    out.local = input.local;
    out.color = input.color;
    return out;
}

@vertex
fn agent_vertex(input: AgentVertexInput) -> VertexOutput {
    let perp = vec2<f32>(-input.dir.y, input.dir.x);
    let world_offset = ((input.dir * input.local.x) + (perp * input.local.y)) * input.radius;
    var out: VertexOutput;
    out.position = vec4<f32>(world_to_ndc(input.pos) + world_delta_to_ndc(world_offset), 0.0, 1.0);
    out.local = input.local;
    out.color = input.color;
    return out;
}

@fragment
fn circle_fragment(input: VertexOutput) -> @location(0) vec4<f32> {
    if dot(input.local, input.local) > 1.0 {
        discard;
    }
    return input.color;
}

@fragment
fn flat_fragment(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::{RenderFoodReadback, RenderSnapshotHeader};

    #[test]
    fn triangle_vertices_normalize_longest_dimension_to_one() {
        let ui_config = UiConfig::default();
        let vertices = triangle_vertices(&ui_config);
        let min_x = vertices.iter().map(|vertex| vertex[0]).fold(f32::INFINITY, f32::min);
        let max_x = vertices.iter().map(|vertex| vertex[0]).fold(f32::NEG_INFINITY, f32::max);
        let min_y = vertices.iter().map(|vertex| vertex[1]).fold(f32::INFINITY, f32::min);
        let max_y = vertices.iter().map(|vertex| vertex[1]).fold(f32::NEG_INFINITY, f32::max);
        let width = max_y - min_y;
        let length = max_x - min_x;

        assert!((length.max(width) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn agent_selection_radius_stays_within_configured_size() {
        let ui_config = UiConfig::default();
        assert!(agent_selection_radius_world(1.0, &ui_config) <= 1.0);
    }

    #[test]
    fn food_size_uses_setting_as_world_diameter() {
        let snapshot = RenderSnapshotReadback {
            header: RenderSnapshotHeader {
                tick: 0,
                total_predators: 0,
                total_prey: 0,
                total_food: 1,
                returned_predators: 0,
                returned_prey: 0,
                returned_food: 1,
            },
            predators: Vec::new(),
            prey: Vec::new(),
            food: vec![RenderFoodReadback { slot: 0, active: 1, reserved0: 0, reserved1: 0, pos_x: 1.0, pos_y: 2.0 }],
        };
        let ui_config = UiConfig { food_size: 1.0, ..UiConfig::default() };
        let frame = WorldFrame::from_snapshot(snapshot, &ui_config);

        assert_eq!(frame.food_instances[0].radius, 0.5);
    }
}
