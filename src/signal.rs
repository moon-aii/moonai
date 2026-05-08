use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};

pub static SHOULD_STOP: AtomicBool = AtomicBool::new(false);

static HANDLER_INIT: Once = Once::new();

pub fn setup_signal_handlers() {
    SHOULD_STOP.store(false, Ordering::SeqCst);
    HANDLER_INIT.call_once(|| {
        let _ = ctrlc::set_handler(|| {
            SHOULD_STOP.store(true, Ordering::SeqCst);
        });
    });
}

pub fn is_signal_pending() -> bool {
    SHOULD_STOP.load(Ordering::SeqCst)
}
