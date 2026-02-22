"""CLI entry point for model-gateway."""

import click


@click.group()
@click.version_option()
def cli():
    """Model Gateway — manage local AI model routing and backends."""


# --- Server lifecycle ---

@cli.command()
@click.option("--port", default=8800, help="Port to listen on.")
@click.option("--config", default="config.yml", help="Path to config file.")
def start(port, config):
    """Start the gateway server."""
    click.echo("Not implemented yet")


@cli.command()
def stop():
    """Stop the running gateway server."""
    click.echo("Not implemented yet")


@cli.command()
def restart():
    """Restart the gateway server."""
    click.echo("Not implemented yet")


# --- Status and info ---

@cli.command()
def status():
    """Show gateway server status."""
    click.echo("Not implemented yet")


@cli.command()
def models():
    """List available models and their backend status."""
    click.echo("Not implemented yet")


# --- Model management ---

@cli.command()
@click.argument("alias")
def switch(alias):
    """Switch the default model to ALIAS."""
    click.echo("Not implemented yet")


@cli.command()
@click.argument("alias")
def test(alias):
    """Send a test prompt to model ALIAS."""
    click.echo("Not implemented yet")


# --- Config management ---

@cli.group()
def config():
    """Manage gateway configuration."""


@config.command("show")
def config_show():
    """Print the current resolved configuration."""
    click.echo("Not implemented yet")


@config.command("edit")
def config_edit():
    """Open the config file in $EDITOR."""
    click.echo("Not implemented yet")


@config.command("validate")
def config_validate():
    """Validate the config file for errors."""
    click.echo("Not implemented yet")


# --- Operations ---

@cli.command()
@click.option("--follow", "-f", is_flag=True, help="Follow log output.")
@click.option("--lines", "-n", default=50, help="Number of lines to show.")
def logs(follow, lines):
    """Show gateway server logs."""
    click.echo("Not implemented yet")


@cli.command()
def health():
    """Check health of all configured backends."""
    click.echo("Not implemented yet")
