import datetime

def get_run_id(tag: str = None) -> str:
    """ Get a unique run ID based on the current date and time, and an optional tag."""
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if tag:
        run_id += f"_{tag}"
    return run_id