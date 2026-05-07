use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub static SHOULD_STOP: AtomicBool = AtomicBool::new(false);

pub fn setup_signal_handlers() {
    let should_stop = Arc::new(AtomicBool::new(false));
    let should_stop_clone = Arc::clone(&should_stop);

    ctrlc::set_handler(move || {
        should_stop_clone.store(true, Ordering::SeqCst);
    })
    .ok();
}

pub fn is_signal_pending() -> bool {
    SHOULD_STOP.load(Ordering::SeqCst)
}
