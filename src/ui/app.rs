use crate::settings::UiConfig;

pub struct App;

impl App {
    pub const fn new(_ui_config: &UiConfig) -> anyhow::Result<Self> {
        Ok(Self)
    }

    pub const fn run(&mut self) {}
}
