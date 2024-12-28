import yaml

def load_config(config_file):
    """Loads configuration from a YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config