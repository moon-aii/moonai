use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rustc-link-lib=dylib=cudart");

    for path in [
        "src/lib.rs",
        "src/config.rs",
        "src/sim",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }

    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let header_path = out_dir.join("moonai_gpu_ffi.hpp");

    let config = cbindgen::Config {
        language: cbindgen::Language::Cxx,
        namespace: Some("moonai_gpu".to_owned()),
        pragma_once: true,
        export: cbindgen::ExportConfig {
            include: vec![
                "CudaStatus".to_owned(),
                "PopulationKind".to_owned(),
                "SimulationConfig".to_owned(),
                "GpuEvolutionConfig".to_owned(),
                "GpuMutationConfig".to_owned(),
                "ReproductionPairReadback".to_owned(),
                "UiStatsReadback".to_owned(),
                "FreeListStateReadback".to_owned(),
                "MetricsSummaryReadback".to_owned(),
                "SensorSnapshotReadback".to_owned(),
                "RenderSnapshotHeader".to_owned(),
                "RenderAgentReadback".to_owned(),
                "RenderFoodReadback".to_owned(),
                "CompiledNetworkReadbackHeader".to_owned(),
                "SelectedAgentNetworkReadback".to_owned(),
                "SpeciesSummaryReadback".to_owned(),
                "RepresentativeGenomeHeader".to_owned(),
                "GenomeNodeReadback".to_owned(),
                "GenomeConnectionReadback".to_owned(),
                "SpeciesBatchReadbackHeader".to_owned(),
                "FoodBuffer".to_owned(),
                "DeviceGenomeBufffers".to_owned(),
                "DeviceCompiledNetworkBuffers".to_owned(),
                "DevicePopulationBuffers".to_owned(),
                "DeviceInnovationState".to_owned(),
                "SimulationCounters".to_owned(),
                "PopulationGridEntry".to_owned(),
                "FoodGridEntry".to_owned(),
                "MetricsReduceScratch".to_owned(),
                "GpuEvolutionState".to_owned(),
            ],
            exclude: vec!["GpuEvolutionStateHandle".to_owned()],
            item_types: vec![cbindgen::ItemType::Enums, cbindgen::ItemType::Structs],
            ..cbindgen::ExportConfig::default()
        },
        ..cbindgen::Config::default()
    };

    cbindgen::Builder::new().with_crate(crate_dir).with_config(config).generate()?.write_to_file(&header_path);

    cc::Build::new()
        .cuda(true)
        .include(&out_dir)
        .flag("-arch=native")
        .flag("-O2")
        .files(&["src/sim/kernel.cu", "src/sim/crossover.cu", "src/sim/mutation.cu", "src/sim/network_compilation.cu"])
        .compile("moonai_cuda");

    Ok(())
}
