import os
import hvac
import logging
from src.utils import load_config
logger = logging.getLogger(__name__)


class VaultClient:
    """
    Wrapper for HashiCorp Vault client (hvac).
    Handles authentication and secret retrieval.
    """

    def __init__(self, addr=None, token=None, token_path="/vault/data/app_token.txt"):
        """
        Args:
            addr (str): Vault server address. Defaults to env VAULT_ADDR or http://vault:8200
            token (str): Vault token. Defaults to env VAULT_TOKEN or read from file.
            token_path (str): Path to token file if env var not set.
        """
        self.addr = addr or os.environ.get("VAULT_ADDR", "http://vault:8200")
        self.token = token or os.environ.get("VAULT_TOKEN", None)

        if not self.token and os.path.exists(token_path):
            try:
                with open(token_path, "r") as f:
                    self.token = f.read().strip()
                    logger.info(f"Loaded Vault token from {token_path}")
            except Exception as e:
                logger.warning(f"Failed to read token from {token_path}: {e}")

        if not self.token:
            logger.warning("No Vault token provided, falling back to 'root' (for testing only).")
            self.token = "root"
        logger.info(f"Connecting to Vault at {self.addr}")
        self.client = hvac.Client(url=self.addr, token=self.token)
        if not self.client.is_authenticated():
            raise RuntimeError("Failed to authenticate with Vault")
        logger.info("Vault client successfully authenticated")

    def get_client(self):
        """Return the underlying hvac.Client"""
        return self.client

    def get_db_credentials(self, path="database/credentials", mount_point="kv"):
        """
        Retrieve database credentials from Vault KV v2.

        Args:
            path (str): Path to the secret in Vault.
            mount_point (str): KV engine mount point.
        
        Returns:
            dict | None: Secret data or None if not found.
        """
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=mount_point
            )
            return response.get("data", {}).get("data")
        except Exception as e:
            logger.error(f"Error retrieving database credentials from Vault: {e}")
            return None
    
    def get_connection(self):
        """
        Get the connection to the Vault server.
        """
        config = load_config("src/config.ini")
        db_config = config["DATABASE"] if "DATABASE" in config else {}
        vault_credentials = self.get_db_credentials(db_config["path"])
        if vault_credentials:
            host = vault_credentials.get("host", "localhost")
            port = int(vault_credentials.get("port", "8123"))
            user = vault_credentials.get("username", "default")
            password = vault_credentials.get("password", "")
            logger.info("Using database credentials from Vault")
        else:
            # Fall back to environment variables
            logger.error("No database credentials found in Vault")
            host = os.environ.get("CLICKHOUSE_HOST", db_config.get("host", "localhost"))
            port = int(os.environ.get("CLICKHOUSE_PORT", "8123"))
            user = os.environ.get("CLICKHOUSE_USER", "default")
            password = os.environ.get("CLICKHOUSE_PASSWORD", "")
            logger.info("Using database credentials from environment variables")
        return host, port, user, password


# Singleton helper
_vault_client = None

def get_vault_client():
    global _vault_client
    if _vault_client is None:
        try:
            _vault_client = VaultClient()
        except Exception as e:
            logger.error(f"Failed to create Vault client: {e}")
            _vault_client = None
    return _vault_client
