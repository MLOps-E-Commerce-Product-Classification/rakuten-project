#!/bin/bash

# --- 1. Environment Configuration ---
echo "Step 1: Configuring .env file..."

# Check if .env.example exists and copy it to .env
if [ -f .env.example ]; then
    cp .env.example .env
    echo "  + Copied .env.example to .env"
else
    echo "  ⚠️  Warning: .env.example not found, creating a new .env"
    touch .env
fi

# Append Airflow and Docker specific configurations
echo -e "\n# Added by setup.sh" >> .env
echo "AIRFLOW_UID=50000" >> .env
echo "AIRFLOW_GID=0" >> .env
echo "DOCKER_GID=$(stat -c '%g' /var/run/docker.sock 2>/dev/null || echo 0)" >> .env
echo "PROJECT_ROOT=$(pwd)" >> .env
echo ".env ready ✅"

# --- 2. DVC Local Config Setup ---
echo "Step 2: Setting up DVC local configuration..."
mkdir -p .dvc
# Extract credentials from the newly created .env
DVC_KEY=$(grep '^DVC_ACCESS_KEY_ID=' .env | cut -d '=' -f2)
DVC_SECRET=$(grep '^DVC_SECRET_ACCESS_KEY=' .env | cut -d '=' -f2)

if [ -n "$DVC_KEY" ] && [ -n "$DVC_SECRET" ]; then
    cat <<EOF > .dvc/config.local
['remote "origin"']
    access_key_id = $DVC_KEY
    secret_access_key = $DVC_SECRET
EOF
    echo "DVC config.local created ✅"
fi

# --- 3. Directory Structure & DVC Pull ---
echo "Step 3: Creating directory structure and pulling data..."
DATA_DIR="./data"
SUBFOLDERS=("new_data" "new_train_data" "new_train_data_archived" "processed" "processed_new" "raw")

for folder in "${SUBFOLDERS[@]}"; do
    mkdir -p "$DATA_DIR/$folder"
done

mkdir -p ./logs ./plugins ./dags

# Pull data from DVC
if command -v uv &> /dev/null; then
    uv run dvc pull
    echo "DVC pull complete ✅"
fi

# --- 4. Final Ownership & Permissions ---
echo "Step 4: Applying final permissions..."

ME=$(whoami)
MY_GROUP=$(id -gn)

# Step A: Clean up ownership (Everything to current user first)
sudo chown -R $ME:$MY_GROUP .

# Step B: Assign specific folders to Airflow (50000)
# These folders must be owned by the container user
sudo chown -R 50000:50000 "$DATA_DIR/new_data" "$DATA_DIR/new_train_data" "$DATA_DIR/new_train_data_archived" "$DATA_DIR/raw"

# Airflow core folders (Logs, Dags, Plugins)
sudo chown -R 50000:$MY_GROUP ./logs ./plugins ./dags

# Step C: Set Permissions
# Full access for Airflow directories to avoid Docker permission issues
sudo chmod -R 777 ./logs ./dags ./plugins

# Standard permissions for the rest
chmod -R 755 "$DATA_DIR"
chmod 775 "$DATA_DIR/new_train_data_archived"

echo "🎉 Setup complete! .env initialized from example and permissions optimized."
