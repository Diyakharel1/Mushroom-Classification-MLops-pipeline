#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Default AIRFLOW_HOME to a local directory if not set
# This helps when running outside Docker without AIRFLOW_HOME explicitly set.
# In Docker, AIRFLOW_HOME is typically set via ENV, overriding this default.
: "${AIRFLOW_HOME:="/app/airflow"}"  # Changed from "./airflow_home" to "/app/airflow"
export AIRFLOW_HOME

# Default AIRFLOW_PORT to 8080 if not set (though we exec with it anyway)
: "${AIRFLOW_PORT:="8080"}"

echo "💨 Initializing folders and permissions for AIRFLOW_HOME: ${AIRFLOW_HOME}"
# Create standard Airflow subdirectories if they don't exist
# This is important because these paths might be volume mounts from the host
# and might not exist or have correct ownership initially.
mkdir -p "${AIRFLOW_HOME}/dags" \
         "${AIRFLOW_HOME}/logs" \
         "${AIRFLOW_HOME}/plugins"

# Enhanced permission check for all critical directories
echo "🔍 Checking permissions for critical directories..."
for dir in "${AIRFLOW_HOME}" "${AIRFLOW_HOME}/dags" "${AIRFLOW_HOME}/logs" "${AIRFLOW_HOME}/plugins"; do
    if [ -w "$dir" ]; then
        echo "✅ Directory $dir is writable"
        
        # Additional check for DAGs directory - can we create Python files?
        if [ "$dir" = "${AIRFLOW_HOME}/dags" ]; then
            if touch "$dir/.test_dag.py" 2>/dev/null; then
                echo "✅ Can create Python files in DAGs directory"
                rm "$dir/.test_dag.py"
            else
                echo "❌ ERROR: Cannot create Python files in $dir"
                echo "    This will prevent DAG discovery from working"
            fi
        fi
    else
        echo "❌ ERROR: Directory $dir is not writable for user $(whoami) (UID $(id -u))"
        echo "    Please run on host: sudo chown -R 50000:50000 $dir"
        echo "    Or: sudo chmod -R 775 $dir"
    fi
done

# Check for existing Python DAG files and their permissions
if [ -d "${AIRFLOW_HOME}/dags" ]; then
    py_files=$(find "${AIRFLOW_HOME}/dags" -name "*.py" 2>/dev/null | wc -l)
    if [ "$py_files" -gt 0 ]; then
        echo "📄 Found $py_files Python files in ${AIRFLOW_HOME}/dags"
        echo "   Sample permissions of Python files:"
        find "${AIRFLOW_HOME}/dags" -name "*.py" -exec ls -l {} \; | head -3
    else
        echo "⚠️ WARNING: No Python DAG files found in ${AIRFLOW_HOME}/dags"
        echo "   - If you expected DAGs, check if files exist in the host directory"
        echo "   - Make sure the correct host directory is mounted to ${AIRFLOW_HOME}/dags"
    fi
fi

# The most likely fix is that the *parent* directory /app/airflow needs to be writable by airflow user
# for it to create subdirs like /app/airflow/logs/scheduler.

# Let's ensure the base logs directory is writable by trying to create a test file.
if touch "${AIRFLOW_HOME}/logs/.airflow_can_write" 2>/dev/null ; then
    echo "✅ Write access to ${AIRFLOW_HOME}/logs confirmed."
    rm "${AIRFLOW_HOME}/logs/.airflow_can_write"
else
    echo "❌ ERROR: No write access to ${AIRFLOW_HOME}/logs for user $(whoami) (UID $(id -u))!"
    echo "This is likely due to host volume mount permissions."
    echo "Please ensure the host directory mounted to ${AIRFLOW_HOME}/logs is writable by UID $(id -u)."
    # As a WAITING strategy to allow manual fix from another terminal:
    # echo "Waiting for 60 seconds to allow manual permission fix on host volumes..."
    # sleep 60
    exit 1 # Exit if we can't write, as Airflow will fail.
fi


# Function to wait for PostgreSQL
wait_for_postgres() {
    echo "Waiting for PostgreSQL at ${POSTGRES_HOST:-localhost}:${POSTGRES_PORT:-5432}..."
    local pg_host="${POSTGRES_HOST:-localhost}"
    local pg_port="${POSTGRES_PORT:-5432}"
    local pg_user="${POSTGRES_USER:-airflow}"
    local pg_db="${POSTGRES_DB:-airflow}"
    local attempt_num=0
    local max_attempts=30
    while ! pg_isready -h "$pg_host" -p "$pg_port" -U "$pg_user" -d "$pg_db" -q; do
        attempt_num=$((attempt_num+1))
        if [ "$attempt_num" -ge "$max_attempts" ]; then
            echo "PostgreSQL did not become ready after $max_attempts attempts. Exiting."
            exit 1
        fi
        echo "PostgreSQL not ready (attempt $attempt_num/$max_attempts), waiting 2 seconds..."
        sleep 2
    done
    echo "✅ PostgreSQL is ready."
}

# Function to wait for Redis
wait_for_redis() {
    echo "Waiting for Redis at ${REDIS_HOST:-localhost}:${REDIS_PORT:-6379}..."
    local redis_host="${REDIS_HOST:-localhost}"
    local redis_port="${REDIS_PORT:-6379}"
    local attempt_num=0
    local max_attempts=30
    while ! redis-cli -h "$redis_host" -p "$redis_port" ping | grep -q "PONG"; do
        attempt_num=$((attempt_num+1))
        if [ "$attempt_num" -ge "$max_attempts" ]; then
            echo "Redis did not become ready after $max_attempts attempts. Exiting."
            exit 1
        fi
        echo "Redis not ready (attempt $attempt_num/$max_attempts), waiting 2 seconds..."
        sleep 2
    done
    echo "✅ Redis is ready."
}


echo "🔁 Waiting for dependencies..."
wait_for_postgres
wait_for_redis

if [ "$1" = "webserver" ] || [ "$1" = "scheduler" ]; then
    # Check DAG directory permissions and visibility
    echo "📁 Checking DAG directory setup..."
    
    # Verify DAG directory exists and has correct permissions
    if [ -d "${AIRFLOW_HOME}/dags" ]; then
        echo "✅ DAG directory exists: ${AIRFLOW_HOME}/dags"
        ls -la "${AIRFLOW_HOME}/dags"
        
        # Check if there are any Python files in the DAGs directory
        if find "${AIRFLOW_HOME}/dags" -name "*.py" | grep -q .; then
            echo "✅ Python DAG files found in ${AIRFLOW_HOME}/dags"
        else
            echo "⚠️ WARNING: No Python files found in ${AIRFLOW_HOME}/dags"
        fi
    else
        echo "❌ ERROR: DAG directory ${AIRFLOW_HOME}/dags does not exist!"
    fi
    
    # Check MLflow connectivity
    echo "🔄 Checking MLflow connectivity..."
    if [ -n "${MLFLOW_TRACKING_URI}" ]; then
        echo "✅ MLFLOW_TRACKING_URI is set to: ${MLFLOW_TRACKING_URI}"
        
        # Extract host and port from MLFLOW_TRACKING_URI
        mlflow_host=$(echo ${MLFLOW_TRACKING_URI} | sed -n 's|http://\([^:]*\):.*|\1|p')
        mlflow_port=$(echo ${MLFLOW_TRACKING_URI} | sed -n 's|http://[^:]*:\([0-9]*\).*|\1|p')
        
        if [ -n "$mlflow_host" ] && [ -n "$mlflow_port" ]; then
            echo "🔍 Testing connection to MLflow server at ${mlflow_host}:${mlflow_port}..."
            # Use timeout to prevent hanging if server is unreachable
            if timeout 5 bash -c "</dev/tcp/${mlflow_host}/${mlflow_port}" 2>/dev/null; then
                echo "✅ MLflow server at ${mlflow_host}:${mlflow_port} is reachable"
            else
                echo "⚠️ WARNING: Cannot connect to MLflow server at ${mlflow_host}:${mlflow_port}"
                echo "   - Please check if MLflow service is running"
                echo "   - Verify network connectivity between containers"
            fi
        else
            echo "⚠️ WARNING: Could not parse MLflow host and port from URI: ${MLFLOW_TRACKING_URI}"
        fi
    else
        echo "⚠️ WARNING: MLFLOW_TRACKING_URI environment variable is not set"
    fi
    
    echo "🔧 Running airflow db migrate..."
    airflow db migrate || { echo "❌ DB migration failed."; exit 1; }
    echo "✅ Database migration successful or already up-to-date."
fi

if [ "$1" = "webserver" ]; then
    echo "👤 Creating/Verifying admin user..."
    if ! airflow users list | grep -q "admin@example.com"; then
        echo "Admin user not found, creating..."
        airflow users create \
          --username admin \
          --firstname Admin \
          --lastname User \
          --role Admin \
          --email admin@example.com \
          --password admin
        echo "✅ Admin user created."
    else
        echo "✅ Admin user 'admin' (admin@example.com) already exists."
    fi
fi

if [ "$1" = "scheduler" ]; then
    echo "🔍 Initializing scheduler with verbose DAG processing..."
    # Print scheduler config for debugging
    echo "Current scheduler configuration:"
    airflow config get-value scheduler dag_dir_list_interval || echo "  - Using default dag_dir_list_interval"
    airflow config get-value scheduler min_file_process_interval || echo "  - Using default min_file_process_interval"
    airflow config get-value core dags_folder || echo "  - Using default dags_folder"
    # Print details on dag_discovery_safe_mode
    airflow config get-value core dag_discovery_safe_mode || echo "  - Using default dag_discovery_safe_mode (True)"
    
    # Verify actual vs expected AIRFLOW_HOME
    echo "🏠 AIRFLOW_HOME verification:"
    echo "  - Expected path: /app/airflow"
    echo "  - Actual environment value: ${AIRFLOW_HOME}"
    echo "  - From airflow config: $(airflow config get-value core airflow_home)"
    
    if [ "${AIRFLOW_HOME}" != "/app/airflow" ]; then
        echo "⚠️ WARNING: AIRFLOW_HOME mismatch! Expected /app/airflow but got ${AIRFLOW_HOME}"
        echo "   This may cause DAG discovery issues. DAGs should be in ${AIRFLOW_HOME}/dags/"
    fi
fi

echo "🚀 Starting Airflow component: $@"
exec airflow "$@"