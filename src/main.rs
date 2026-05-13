use anyhow::{Context as _, Result, bail};
use moonai::experiment::ExperimentCatalog;
use moonai::settings;
use moonai::sim::{CudaStatus, runtime_status};
use moonai::ui::app::App;

fn main() -> Result<()> {
    let root_dir = std::env::current_exe()
        .context("failed to resolve current executable path")?
        .parent()
        .context("failed to resolve executable directory")?
        .to_path_buf();

    let settings = settings::load_settings(&root_dir)?;
    let experiments = ExperimentCatalog::load(&root_dir)?;

    let cuda_status = runtime_status();
    if cuda_status != CudaStatus::Success {
        bail!("CUDA runtime unavailable: {cuda_status:?}");
    }

    App::run(&root_dir, experiments, settings)
}
