#!/bin/sh

# Start Vault server in dev mode
vault server -dev -dev-root-token-id=root -dev-listen-address=0.0.0.0:8200 &

# Wait for Vault to start
sleep 5

# Run the initialization script if Vault is not already initialized
if [ ! -f /vault/data/init.txt ]; then
  echo "Initializing Vault..."
  /vault/init.sh
else
  echo "Vault already initialized. Configuring..."
  /vault/configure.sh
fi

# Keep the container running
tail -f /dev/null