#!/bin/bash

# --- 1. Environment Configuration ---
echo "Step 1: Configuring .env file..."

# Append Airflow and Docker specific configurations
echo -e "\n# Added by setup.sh" >> .env
echo "AIRFLOW_UID=50000" >> .env
echo "AIRFLOW_GID=0" >> .env
echo "DOCKER_GID=$(stat -c '%g' /var/run/docker.sock 2>/dev/null || echo 0)" >> .env
echo "PROJECT_ROOT=$(pwd)" >> .env
echo ".env ready :white_check_mark:"

# --- 2. DVC Local Config Setup ---
echo "Step 2: Setting up DVC local configuration..."
mkdir -p .dvc

# Extract credentials from the newly created .env
DVC_KEY=$(grep '^TOKEN=' .env | cut -d '=' -f2)
DVC_SECRET=$(grep '^TOKEN=' .env | cut -d '=' -f2)

if [ -n "$DVC_KEY" ] && [ -n "$DVC_SECRET" ]; then
    cat <<EOF > .dvc/config.local
['remote "origin"']
    access_key_id = $DVC_KEY
    secret_access_key = $DVC_SECRET
EOF
    echo "  + DVC config.local created :white_check_mark:"
else
    echo "  :warning:  Note: DVC credentials missing in .env, skipping config.local creation."
fi

# --- 3. Directory Structure, Sync & DVC Pull ---
echo "Step 3: Creating directory structure and pulling data..."
DATA_DIR="./data"
SUBFOLDERS=("new_data" "new_train_data" "new_train_data_archived" "processed" "processed_new" "raw")

for folder in "${SUBFOLDERS[@]}"; do
    mkdir -p "$DATA_DIR/$folder"
done

mkdir -p ./logs ./plugins ./dags

# Pull data from DVC
if command -v uv &> /dev/null; then
    echo "  Installing all dependencies (dev group & extras)..."
    uv sync --all-extras --group dev --link-mode copy

    echo "  Running dvc pull..."
    uv run dvc pull
    echo "DVC pull complete :white_check_mark:"
else
    echo ":x: Error: 'uv' is not installed. Skipping DVC pull."
fi

# --- 4. Final Ownership & Permissions ---
echo "Step 4: Applying final permissions..."

ME=$(whoami)
MY_GROUP=$(id -gn)

# Step A: Clean up ownership (Everything to current user first)
# This ensures files created by 'uv' or 'root' belong to you
sudo chown -R $ME:$MY_GROUP .

# Step B: Assign specific folders to Airflow (50000)
echo "  Setting Airflow ownership for specific data and core folders..."
sudo chown -R 50000:50000 "$DATA_DIR/new_data" "$DATA_DIR/new_train_data" "$DATA_DIR/new_train_data_archived" "$DATA_DIR/raw"
sudo chown -R 50000:$MY_GROUP ./logs ./plugins ./dags

# Step C: Set Permissions
# 777 for Airflow core folders to prevent any Docker mount issues
sudo chmod -R 777 ./logs ./dags ./plugins

# Standard permissions for data (755) and special case for archived (775)
sudo chmod -R 755 "$DATA_DIR"
sudo chmod 775 "$DATA_DIR/new_train_data_archived"

echo "Permissions and ownership set :white_check_mark:"
echo ":tada: Setup complete! All systems go."
