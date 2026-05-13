use anyhow::{Context as _, Result};
use moonai::experiment::ExperimentCatalog;
use moonai::settings;
use moonai::ui::app::App;

fn main() -> Result<()> {
    let root_dir = std::env::current_exe()
        .context("failed to resolve current executable path")?
        .parent()
        .context("failed to resolve executable directory")?
        .to_path_buf();

    let settings = settings::load_settings(&root_dir)?;
    let experiments = ExperimentCatalog::load(&root_dir)?;

    App::run(&root_dir, experiments, settings)
}
