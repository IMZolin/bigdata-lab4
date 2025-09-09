import os
import hvac
import configparser
from src.logger import Logger
from src.database import ClickHouseClient

SHOW_LOG = True

class VaultClient:
    """
    Wrapper for HashiCorp Vault client (hvac).
    Handles authentication and secret retrieval.
    """

    def __init__(self, addr=None, token=None, token_path="/vault/data/app_token.txt", config_path="config.ini"):
        """
        Args:
            addr (str): Vault server address. Defaults to env VAULT_ADDR or http://vault:8200
            token (str): Vault token. Defaults to env VAULT_TOKEN or read from file.
            token_path (str): Path to token file if env var not set.
        """
        self.addr = addr or os.environ.get("VAULT_ADDR", "http://vault:8200")
        self.token = token or os.environ.get("VAULT_TOKEN", None)
        self.logger = Logger(show=SHOW_LOG).get_logger(__name__)
        self.config = configparser.ConfigParser()
        self.config_path = config_path
        self.config.read(config_path)

        if not self.token and os.path.exists(token_path):
            try:
                with open(token_path, "r") as f:
                    self.token = f.read().strip()
                    self.logger.info(f"Loaded Vault token from {token_path}")
            except Exception as e:
                self.logger.warning(f"Failed to read token from {token_path}: {e}")

        if not self.token:
            self.logger.warning("No Vault token provided, falling back to 'root' (for testing only).")
            self.token = "root"
        self.logger.info(f"Connecting to Vault at {self.addr}")
        self.client = hvac.Client(url=self.addr, token=self.token)
        if not self.client.is_authenticated():
            raise RuntimeError("Failed to authenticate with Vault")
        self.logger.info("Vault client successfully authenticated")

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
            self.logger.error(f"Error retrieving database credentials from Vault: {e}")
            return None

    def is_authenticated(self):
        return self.client and self.client.is_authenticated()

    def list_mounted_secrets_engines(self):
        return self.client and self.client.sys.list_mounted_secrets_engines()
    
    def get_connection(self):
        """
        Get the connection to the Vault server.
        """
        db_config = self.config["DATABASE"] if "DATABASE" in self.config else {}
        path = db_config.get("path", "database/credentials")
        mount_point = db_config.get("mount_point", "kv")
        self.logger.info(f"Fetching Vault credentials from path='{path}', mount_point='{mount_point}'")
        vault_credentials = self.get_db_credentials(path=path, mount_point=mount_point)
        if vault_credentials:
            host = vault_credentials.get("host", db_config.get("host", "clickhouse"))
            port = int(vault_credentials.get("port", db_config.get("port", "8123")))
            user = vault_credentials.get("username", db_config.get("user", "default"))
            password = vault_credentials.get("password", db_config.get("password", ""))
            self.logger.info("Using database credentials from Vault")
            self.logger.info(f"{host}, {port}, {user}, {password}")
        else:
            # Fall back to environment variables
            host = os.environ.get("CLICKHOUSE_HOST", db_config.get("host", "clickhouse"))
            port = int(os.environ.get("CLICKHOUSE_PORT", db_config.get("port", "8123")))
            user = os.environ.get("CLICKHOUSE_USER", db_config.get("user", "default"))
            password = os.environ.get("CLICKHOUSE_PASSWORD", db_config.get("password", ""))
            self.logger.info("Using database credentials from environment variables")
        return host, port, user, password

    def setup_database(self, table_name="predictions", max_attempts=5):
        """
        Setup ClickHouse client using credentials from Vault.
        Returns:
            ClickHouseClient object if successful, else None.
        """
        for attempt in range(max_attempts):
            try:
                host, port, user, password = self.get_connection()
                db_client = ClickHouseClient(host, port, user, password)
                db_client.connect()
                db_client.create_table(table_name)
                self.logger.info(f"Database setup completed successfully (table: {table_name})")
                return db_client
            except Exception as e:
                self.logger.error(f"Database setup failed (attempt {attempt+1}/{max_attempts}): {e}")
                time.sleep(3)
        self.logger.warning(f"Failed to setup database after {max_attempts} attempts")
        return None



# Singleton helper
_vault_client = None

def get_vault_client():
    logger = Logger(show=SHOW_LOG).get_logger(__name__)
    global _vault_client
    if _vault_client is None:
        try:
            _vault_client = VaultClient()
        except Exception as e:
            logger.error(f"Failed to create Vault client: {e}")
            _vault_client = None
    return _vault_client
