import click

from .runtime import configure_logging
from .lightning_pipeline import run_date_range


@click.command()
@click.option(
    '--lightning-dir', 'lightning_base_path',
    type=click.Path(file_okay=False),
    required=True,
    help="Base directory for Lightning downloads and outputs"
)
@click.option(
    '--log-dir',
    type=click.Path(file_okay=False),
    default='logs',
    show_default=True,
    help="Directory where monthly logs are written"
)
@click.option(
    '--start-date',
    type=click.DateTime(formats=['%Y-%m-%d']),
    required=True,
    help="Start date (YYYY-MM-DD)"
)
@click.option(
    '--end-date',
    type=click.DateTime(formats=['%Y-%m-%d']),
    required=True,
    help="End date (YYYY-MM-DD)"
)
@click.option(
    '--product', 'products',
    multiple=True,
    default=['MSI_COP_2A', 'CPR_FMR_2A'],
    show_default=True,
    help="EarthCARE products to process"
)
@click.option(
    '--frame', 'frames',
    multiple=True,
    default=['A', 'B', 'D', 'E', 'F', 'H'],
    show_default=True,
    help="Orbit frames to include"
)
@click.option(
    '--half-window', 'half_window_minutes',
    default=60,
    show_default=True,
    help="Half-window of LI integration in minutes"
)
@click.option(
    '--lightning-platform', 'lightning_platforms',
    multiple=True,
    type=click.Choice(['MTG-I1', 'GOES-16', 'GOES-18', 'GOES-19']),
    default=['MTG-I1', 'GOES-16', 'GOES-18', 'GOES-19'],
    show_default=True,
    help="Lightning platforms to include"
)
@click.option(
    '--distance-threshold', 'distance_threshold_km',
    type=float,
    default=5,
    show_default=True,
    help="Distance threshold from CPR track in kilometers"
)
@click.option(
    '--time-threshold', 'time_threshold_s',
    type=int,
    default=300,
    show_default=True,
    help="Temporal threshold from CPR acquisition in seconds"
)
def run_pipeline(
    lightning_base_path,
    log_dir,
    start_date,
    end_date,
    products,
    frames,
    half_window_minutes,
    lightning_platforms,
    distance_threshold_km,
    time_threshold_s
):
    """Run the EarthCARE-lightning collocation pipeline over a date range."""
    configure_logging()

    run_date_range(
        lightning_base_path=lightning_base_path,
        log_dir=log_dir,
        start_date=start_date,
        end_date=end_date,
        products=products,
        frames=frames,
        half_window_minutes=half_window_minutes,
        lightning_platforms=lightning_platforms,
        distance_threshold_km=distance_threshold_km,
        time_threshold_s=time_threshold_s,
    )


def main():
    run_pipeline()


if __name__ == "__main__":
    main()