import yaml
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

def load_config():
    """Loads the YAML configuration file."""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Replace "today" with the current date
    if config['data_fetching']['end_date'] == 'today':
        config['data_fetching']['end_date'] = datetime.now().strftime('%Y-%m-%d')
        
    return config

def get_api_key(service_name: str):
    """
    Retrieves a specific API key from the environment variable first,
    or falls back to the config file.
    
    Args:
        service_name: The name of the service (e.g., 'polygon', 'newsapi').
    
    Returns:
        The API key.
    """
    # Environment variable names are typically uppercase
    env_var = f"{service_name.upper()}_API_KEY"
    
    # First, try to get the key from an environment variable
    api_key = os.getenv(env_var)
    if api_key:
        return api_key
        
    # If not found, fall back to the config file
    config = load_config()
    api_key = config['api_keys'].get(service_name)
    
    if not api_key or api_key == "YOUR_API_KEY":
        raise ValueError(f"API key for {service_name} not found. "
                         f"Please set the {env_var} environment variable or "
                         f"update the 'dags/config.yaml' file.")
    
    return api_key

if __name__ == '__main__':
    # Example of how to use these functions
    config = load_config()
    print("--- Configuration ---")
    print(config)
    
    print("\n--- API Keys ---")
    for service in ['polygon', 'newsapi', 'alpha_vantage', 'fred']:
        try:
            key = get_api_key(service)
            print(f"{service.title()} Key: {key[:4]}...{key[-4:]}") # Print a masked version
        except ValueError as e:
            print(f"{service.title()} Key: {e}") 