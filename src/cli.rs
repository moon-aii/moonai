use clap::Parser;

#[derive(Debug, Parser)]
#[command(author, version, about)]
pub struct CliArgs {
    #[arg(short, long)]
    pub config: Option<String>,
    #[arg(long)]
    pub settings: Option<String>,
    #[arg(short = 'n', long)]
    pub ticks: Option<i32>,
    #[arg(long)]
    pub headless: bool,
    #[arg(short, long)]
    pub verbose: bool,
    #[arg(long)]
    pub experiment: Option<String>,
    #[arg(long)]
    pub all: bool,
    #[arg(long)]
    pub list: bool,
    #[arg(long)]
    pub name: Option<String>,
    #[arg(long)]
    pub validate: bool,
}
