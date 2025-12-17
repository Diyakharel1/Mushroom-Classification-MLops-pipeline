#!/bin/bash
set -e

# 🐛 DEBUG: Print all arguments received by the script
echo "🐛 DEBUG: Script called with $# arguments:"
echo "🐛 DEBUG: All arguments: $*"
echo "🐛 DEBUG: Arguments array: $@"
for i in $(seq 1 $#); do
    eval "arg=\${$i}"
    echo "🐛 DEBUG: Argument $i: '$arg'"
done
echo "🐛 DEBUG: First argument (\$1): '$1'"
echo "🐛 DEBUG: Second argument (\$2): '$2'"
echo "🐛 DEBUG: Third argument (\$3): '$3'"
echo "============================================================"

echo "🐳 Starting Mushroom Classification MLOps Pipeline"
echo "============================================================"

# Debug: Print all arguments received by the entrypoint script
echo "🐛 DEBUG: Entrypoint script received ${#} arguments:"
for i in $(seq 1 $#); do
    echo "  Arg $i: '${!i}'"
done
echo "🐛 DEBUG: All args as string: '$*'"
echo "🐛 DEBUG: All args as array: '$@'"
echo "============================================================"

# Detect if running in Docker or on host
if [ -f "/.dockerenv" ]; then
    echo "🐳 Running inside Docker container"
    PROJECT_DIR="/app"
    DATA_DIR="/app/data"
    AIRFLOW_HOME="/app/airflow"
else
    echo "🖥️ Running on host system"
    PROJECT_DIR="$(pwd)"
    DATA_DIR="$(pwd)/data"
    AIRFLOW_HOME="$(pwd)/airflow"
    
    echo "💡 To run in Docker instead:"
    echo "   docker build -t mushroom-mlops ."
    echo "   docker run -it --rm -p 5000:5000 -p 8080:8080 mushroom-mlops"
    echo ""
fi

# Function to show Docker management commands
show_docker_commands() {
    echo "🐳 Docker Management Commands:"
    echo ""
    echo "1. Clean up old images:"
    echo "   docker system prune -a"
    echo "   docker image prune -a"
    echo ""
    echo "2. Build fresh image:"
    echo "   docker build -t mushroom-mlops ."
    echo ""
    echo "3. Run standalone container:"
    echo "   docker run -it --rm -p 5000:5000 -p 8080:8080 \\
         -v \$(pwd)/data:/app/data \\
         -v \$(pwd)/logs:/app/logs \\
         mushroom-mlops"
    echo ""
    echo "4. Run in background:"
    echo "   docker run -d --name mushroom-pipeline \\
         -p 5000:5000 -p 8080:8080 \\
         -v \$(pwd)/data:/app/data \\
         -v \$(pwd)/logs:/app/logs \\
         mushroom-mlops"
    echo ""
    echo "5. View logs:"
    echo "   docker logs -f mushroom-pipeline"
    echo ""
    echo "6. Stop container:"
    echo "   docker stop mushroom-pipeline"
    echo "   docker rm mushroom-pipeline"
    echo ""
    echo "🖥️ Host System Commands:"
    echo "   ./run_simple.sh run-all    # Run with existing script"
    echo "   ./docker-entrypoint.sh local # Run locally"
}

# Function to check if service is running
check_service() {
    local service_name=$1
    local port=$2
    local max_attempts=30
    local attempt=1
    
    echo "🔍 Checking $service_name on port $port..."
    while [ $attempt -le $max_attempts ]; do
        # Use different methods based on environment
        if command -v netstat >/dev/null 2>&1; then
            if netstat -tuln 2>/dev/null | grep -q ":$port "; then
                echo "✅ $service_name is ready!"
                return 0
            fi
        elif command -v ss >/dev/null 2>&1; then
            if ss -tuln 2>/dev/null | grep -q ":$port "; then
                echo "✅ $service_name is ready!"
                return 0
            fi
        else
            # Fallback: try to connect
            if timeout 1 bash -c "</dev/tcp/localhost/$port" 2>/dev/null; then
                echo "✅ $service_name is ready!"
                return 0
            fi
        fi
        echo "⏳ Waiting for $service_name... (attempt $attempt/$max_attempts)"
        sleep 2
        attempt=$((attempt + 1))
    done
    echo "❌ $service_name failed to start"
    return 1
}

# Function to create sample data if needed
create_sample_data() {
    echo "📊 Creating sample data..."
    python3 -c "
import pandas as pd
import numpy as np
import os

# Use environment-specific paths
project_dir = '$PROJECT_DIR'
data_dir = '$DATA_DIR'

# Create sample mushroom data
np.random.seed(42)
n_samples = 1000

sample_data = pd.DataFrame({
    'class': np.random.choice(['e', 'p'], n_samples),
    'cap_diameter': np.abs(np.random.normal(50, 20, n_samples)),
    'stem_height': np.abs(np.random.normal(100, 30, n_samples)),
    'stem_width': np.abs(np.random.normal(10, 3, n_samples)),
    'cap_shape': np.random.choice(['bell', 'conical', 'flat'], n_samples),
    'cap_color': np.random.choice(['brown', 'white', 'red'], n_samples),
    'habitat': np.random.choice(['forest', 'urban', 'meadow'], n_samples),
    'does_bruise_or_bleed': np.random.choice(['t', 'f'], n_samples),
    'has_ring': np.random.choice(['t', 'f'], n_samples),
    'gill_color': np.random.choice(['brown', 'white', 'black'], n_samples),
    'stem_color': np.random.choice(['brown', 'white', 'yellow'], n_samples)
})

# Save sample data
raw_data_dir = os.path.join(data_dir, 'raw')
os.makedirs(raw_data_dir, exist_ok=True)
sample_data_path = os.path.join(raw_data_dir, 'sample_data.csv')
sample_data.to_csv(sample_data_path, index=False)
print(f'✅ Sample data created with {len(sample_data)} records at {sample_data_path}')
"
}

# Function to create a simple Airflow DAG
create_mushroom_dag() {
    echo "📋 Creating Mushroom Classification DAG..."
    mkdir -p "$AIRFLOW_HOME/dags"
    
    cat > "$AIRFLOW_HOME/dags/mushroom_classification_dag.py" << EOF
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

default_args = {
    'owner': 'mushroom-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'mushroom_classification_pipeline',
    default_args=default_args,
    description='Mushroom Classification MLOps Pipeline',
    schedule_interval=timedelta(hours=6),
    catchup=False,
    tags=['mlops', 'mushroom', 'classification'],
)

def extract_data(**context):
    """Extract and prepare mushroom data"""
    print("🔍 Extracting mushroom data...")
    
    # Use environment-appropriate paths
    data_dir = '$DATA_DIR'
    data_path = os.path.join(data_dir, 'raw', 'sample_data.csv')
    
    if not os.path.exists(data_path):
        np.random.seed(42)
        n_samples = 1000
        
        sample_data = pd.DataFrame({
            'class': np.random.choice(['e', 'p'], n_samples),
            'cap_diameter': np.abs(np.random.normal(50, 20, n_samples)),
            'stem_height': np.abs(np.random.normal(100, 30, n_samples)),
            'stem_width': np.abs(np.random.normal(10, 3, n_samples)),
            'cap_shape': np.random.choice(['bell', 'conical', 'flat'], n_samples),
            'cap_color': np.random.choice(['brown', 'white', 'red'], n_samples),
        })
        
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        sample_data.to_csv(data_path, index=False)
    
    df = pd.read_csv(data_path)
    print(f"✅ Data extracted: {df.shape}")
    return data_path

def train_model(**context):
    """Train mushroom classification model"""
    print("🤖 Training mushroom classification model...")
    
    # Set MLflow tracking
    mlflow.set_tracking_uri('http://localhost:5000')
    
    with mlflow.start_run(run_name=f"airflow_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Load data
        data_path = context['task_instance'].xcom_pull(task_ids='extract_data')
        df = pd.read_csv(data_path)
        
        # Simple preprocessing
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(df['class'])
        
        numerical_features = ['cap_diameter', 'stem_height', 'stem_width']
        X = df[numerical_features].fillna(df[numerical_features].mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log to MLflow
        mlflow.log_param('model_type', 'RandomForest')
        mlflow.log_param('n_estimators', 100)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.sklearn.log_model(model, 'model')
        
        print(f"✅ Model trained with accuracy: {accuracy:.4f}")
        return accuracy

# Define tasks
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

health_check_task = BashOperator(
    task_id='health_check',
    bash_command='echo "🔍 Pipeline health check completed" && echo "MLflow: \$(curl -s http://localhost:5000/health || echo "offline")"',
    dag=dag,
)

# Set task dependencies
extract_task >> train_task >> health_check_task
EOF

    echo "✅ Mushroom Classification DAG created"
}

# Function to check pipeline status
check_pipeline_status() {
    echo "🔍 Checking MLOps Pipeline Status"
    echo "============================================================"
    
    # Check MLflow
    echo "📊 MLflow Status:"
    if curl -s "http://localhost:5000" > /dev/null 2>&1; then
        echo "   ✅ MLflow UI: http://localhost:5000 (RUNNING)"
        # Try to get experiment info
        curl -s "http://localhost:5000/api/2.0/mlflow/experiments/search" > /dev/null 2>&1 && echo "   ✅ MLflow API: Responding" || echo "   ⚠️ MLflow API: Not responding"
    else
        echo "   ❌ MLflow UI: http://localhost:5000 (DOWN)"
    fi
    
    # Check Airflow
    echo ""
    echo "🔄 Airflow Status:"
    if curl -s "http://localhost:8080" > /dev/null 2>&1; then
        echo "   ✅ Airflow UI: http://localhost:8080 (RUNNING)"
        echo "   👤 Login: admin/admin"
    else
        echo "   ❌ Airflow UI: http://localhost:8080 (DOWN)"
    fi
    
    # Check if processes are running
    echo ""
    echo "🔧 Process Status:"
    if pgrep -f "mlflow server" > /dev/null; then
        mlflow_pid=$(pgrep -f "mlflow server")
        echo "   ✅ MLflow server: PID $mlflow_pid"
    else
        echo "   ❌ MLflow server: Not running"
    fi
    
    if pgrep -f "airflow webserver" > /dev/null; then
        airflow_web_pid=$(pgrep -f "airflow webserver")
        echo "   ✅ Airflow webserver: PID $airflow_web_pid"
    else
        echo "   ❌ Airflow webserver: Not running"
    fi
    
    if pgrep -f "airflow scheduler" > /dev/null; then
        airflow_sch_pid=$(pgrep -f "airflow scheduler")
        echo "   ✅ Airflow scheduler: PID $airflow_sch_pid"
    else
        echo "   ❌ Airflow scheduler: Not running"
    fi
    
    # Check data directory
    echo ""
    echo "📁 Data Status:"
    if [ -d "$DATA_DIR/raw" ]; then
        data_files=$(find "$DATA_DIR/raw" -name "*.csv" | wc -l)
        echo "   ✅ Data directory: $DATA_DIR/raw ($data_files CSV files)"
    else
        echo "   ❌ Data directory: $DATA_DIR/raw (Missing)"
    fi
    
    # Check Airflow DAGs
    echo ""
    echo "📋 Airflow DAGs:"
    if [ -f "$AIRFLOW_HOME/dags/mushroom_classification_dag.py" ]; then
        echo "   ✅ Mushroom Classification DAG: Available"
    else
        echo "   ❌ Mushroom Classification DAG: Missing"
    fi
    
    echo ""
    echo "💡 Quick Commands:"
    echo "   Start Airflow: ./docker-entrypoint.sh start-airflow"
    echo "   Start MLflow: ./docker-entrypoint.sh start-mlflow"
    echo "   Start all: ./docker-entrypoint.sh local"
    echo "   Check status: ./docker-entrypoint.sh status"
    echo "   Stop all: ./docker-entrypoint.sh stop"
    echo "   Docker (alt ports): docker run -it --rm -p 5001:5000 -p 8081:8080 mushroom-mlops"
}

# Function to start only Airflow
start_airflow_only() {
    echo "🔄 Starting Airflow services only..."
    
    # Setup Airflow
    setup_airflow
    
    # Create DAG
    create_mushroom_dag
    
    # Start Airflow scheduler in background
    echo "📅 Starting Airflow scheduler..."
    airflow scheduler &
    SCHEDULER_PID=$!
    
    # Start Airflow webserver in background
    echo "🌐 Starting Airflow webserver..."
    airflow webserver --port 8080 --daemon &
    WEBSERVER_PID=$!
    
    # Wait for services to start
    echo "⏳ Waiting for Airflow to start..."
    sleep 15
    
    # Check status
    check_pipeline_status
    
    echo ""
    echo "✅ Airflow started! Access at: http://localhost:8080 (admin/admin)"
}

# Function to start only MLflow
start_mlflow_only() {
    echo "📊 Starting MLflow server only..."
    
    # Check if MLflow is already running
    if pgrep -f "mlflow server" > /dev/null; then
        echo "⚠️ MLflow is already running"
        return 0
    fi
    
    # Start MLflow server
    mlflow server \
        --host 0.0.0.0 \
        --port 5000 \
        --backend-store-uri sqlite:///$PROJECT_DIR/mlflow.db \
        --default-artifact-root $PROJECT_DIR/mlruns &
    MLFLOW_PID=$!
    
    echo "⏳ Waiting for MLflow to start..."
    sleep 10
    
    # Check status
    check_pipeline_status
    
    echo ""
    echo "✅ MLflow started! Access at: http://localhost:5000"
}

# Function to run training with improved error handling
run_training() {
    echo "🧠 Running training pipeline..."
    
    # Use environment-specific data path
    data_path="$DATA_DIR/raw/sample_data.csv"
    
    python3 -c "
import sys
import os
import time
sys.path.append('$PROJECT_DIR')

# Wait a bit for services
time.sleep(5)

try:
    # Import required modules
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import LabelEncoder
    import mlflow
    import mlflow.sklearn
    from datetime import datetime
    
    print('📦 All imports successful')
    
    # Set MLflow tracking
    mlflow.set_tracking_uri('http://localhost:5000')
    
    # Create experiment
    experiment_name = 'mushroom-classification-local'
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f'✅ Created experiment: {experiment_name}')
    except:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id if experiment else '0'
        print(f'ℹ️ Using existing experiment: {experiment_name}')
    
    # Load data
    data_path = '$data_path'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f'📊 Data loaded: {df.shape}')
    else:
        print('❌ No data file found')
        sys.exit(1)
    
    # Simple preprocessing
    print('🔄 Preprocessing data...')
    
    # Encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(df['class'])
    
    # Select numerical features
    numerical_features = ['cap_diameter', 'stem_height', 'stem_width']
    X_num = df[numerical_features].fillna(df[numerical_features].mean())
    
    # Encode categorical features
    categorical_features = [col for col in ['cap_shape', 'cap_color', 'habitat', 'does_bruise_or_bleed', 'has_ring'] if col in df.columns]
    if categorical_features:
        X_cat = pd.get_dummies(df[categorical_features], drop_first=True)
        X = pd.concat([X_num, X_cat], axis=1)
    else:
        X = X_num
    
    print(f'✅ Features prepared: {X.shape}')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models with MLflow tracking
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    best_model_name = None
    best_accuracy = 0
    
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f'{model_name}_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}', experiment_id=experiment_id):
            print(f'🤖 Training {model_name}...')
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log parameters and metrics
            mlflow.log_param('model_type', model_name)
            mlflow.log_param('n_features', X.shape[1])
            mlflow.log_param('train_size', len(X_train))
            mlflow.log_param('test_size', len(X_test))
            mlflow.log_metric('accuracy', accuracy)
            
            # Log model
            mlflow.sklearn.log_model(model, 'model')
            
            print(f'✅ {model_name} accuracy: {accuracy:.4f}')
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = model_name
    
    print(f'\\n🏆 Best model: {best_model_name} with accuracy: {best_accuracy:.4f}')
    print('✅ Training completed successfully!')

except ImportError as e:
    print(f'❌ Import error: {e}')
    print('Make sure you have the required packages installed in your environment')
except Exception as e:
    print(f'❌ Training failed: {e}')
    import traceback
    traceback.print_exc()
"
}

# Wait for dependencies
echo "⏳ Waiting for dependencies..."
sleep 5

# Create sample data
create_sample_data

# Start services based on command
echo "🐛 DEBUG: About to enter case statement with first arg: '$1'"
case "${1:-run-all}" in
    "status")
        check_pipeline_status
        ;;
        
    "stop")
        stop_pipeline
        ;;
        
    "restart")
        echo "🔄 Restarting MLOps Pipeline..."
        stop_pipeline
        sleep 3
        ${0} local
        ;;

    "reset-airflow")
        echo "🔑 Resetting Airflow credentials..."
        reset_airflow_credentials
        ;;

    "start-airflow")
        start_airflow_only
        ;;
        
    "start-mlflow")
        start_mlflow_only
        ;;

    "local")
        echo "🖥️ Running locally (not in Docker)..."
        echo "This will use your local Python environment and ./run_simple.sh"
        echo ""
        if [ -f "./run_simple.sh" ]; then
            ./run_simple.sh run-all
        else
            echo "❌ run_simple.sh not found. Please ensure you're in the project directory."
            exit 1
        fi
        ;;
        
    "docker-help")
        show_docker_commands
        ;;

    "cleanup")
        echo "🧹 Cleaning up Docker environment..."
        echo "This will remove unused images and containers."
        show_docker_commands
        ;;

    "run-all")
        if [ ! -f "/.dockerenv" ]; then
            echo "⚠️ You're running this on the host system, not in Docker."
            echo "💡 Did you mean to run:"
            echo "   ./docker-entrypoint.sh local     # Run locally"
            echo "   ./docker-entrypoint.sh start-airflow  # Start just Airflow"
            echo "   docker run -it --rm -p 5001:5000 -p 8081:8080 mushroom-mlops  # Docker with alt ports"
            echo ""
            echo "🤔 Continue with local execution? [y/N]"
            read -r response
            if [[ ! "$response" =~ ^[Yy]$ ]]; then
                echo "Exiting. Use 'status' to check what's running."
                exit 0
            fi
        fi
        
        echo "🚀 Starting all services..."
        
        # Setup Airflow
        setup_airflow
        
        # Create DAG
        create_mushroom_dag
        
        # Start MLflow server in background (only if not running)
        if ! pgrep -f "mlflow server" > /dev/null; then
            echo "📊 Starting MLflow server..."
            mlflow server \
                --host 0.0.0.0 \
                --port 5000 \
                --backend-store-uri sqlite:///$PROJECT_DIR/mlflow.db \
                --default-artifact-root $PROJECT_DIR/mlruns &
            MLFLOW_PID=$!
        else
            echo "📊 MLflow already running"
        fi
        
        # Start Airflow scheduler in background
        echo "📅 Starting Airflow scheduler..."
        airflow scheduler &
        SCHEDULER_PID=$!
        
        # Start Airflow webserver in background
        echo "🌐 Starting Airflow webserver..."
        airflow webserver --port 8080 --daemon &
        WEBSERVER_PID=$!
        
        # Wait for services to start
        echo "⏳ Waiting for services to start..."
        sleep 20
        
        # Check if services are running
        check_pipeline_status
        
        # Run training pipeline
        run_training
        
        echo ""
        echo "✅ Pipeline started! Use these commands:"
        echo "   ./docker-entrypoint.sh status   # Check status"
        echo "   ./docker-entrypoint.sh stop     # Stop all services"
        echo "   ./docker-entrypoint.sh restart  # Restart services"
        ;;

    "docker-entrypoint.sh")
        echo "🔄 Running entrypoint script directly..."
        exec "$0" run-all
        ;;

    "mlflow-only")
        echo "📊 Starting MLflow server only..."
        exec mlflow server \
            --host 0.0.0.0 \
            --port 5000 \
            --backend-store-uri sqlite:///$PROJECT_DIR/mlflow.db \
            --default-artifact-root $PROJECT_DIR/mlruns
        ;;
        
    "airflow-only")
        echo "🔄 Starting Airflow only..."
        setup_airflow
        airflow scheduler &
        exec airflow webserver --port 8080
        ;;
        
    "train-only")
        echo "🧠 Running training only..."
        create_sample_data
        run_training
        ;;
        
    "bash")
        # Support for bash as a command
        echo "🐚 Running bash shell..."
        exec bash "${@:2}"
        ;;
        
    "-c")
        # Support for using -c option directly
        echo "🐚 Executing bash command..."
        exec bash -c "${*:2}"
        ;;
        
    *)
        # Debug: Print what we're checking in the wildcard case
        echo "🐛 DEBUG: Wildcard case triggered"
        echo "🐛 DEBUG: \$1 = '$1'"
        echo "🐛 DEBUG: \$2 = '$2'"
        echo "🐛 DEBUG: \${*:3} = '${*:3}'"
        
        # Check if the first argument is bash with -c
        if [[ "$1" == "bash" && "$2" == "-c" ]]; then
            echo "🐚 Executing bash command..."
            echo "🐛 DEBUG: Command to execute: '${*:3}'"
            exec bash -c "${*:3}"
        else
            echo "🐛 DEBUG: Condition failed - not 'bash -c'"
            echo "❌ Unknown command: $1"
            echo "Available commands:"
            echo "  status          - Check pipeline status"
            echo "  reset-airflow   - Reset Airflow credentials"
            echo "  start-airflow   - Start only Airflow"
            echo "  start-mlflow    - Start only MLflow"
            echo "  local           - Run with existing run_simple.sh"
            echo "  run-all         - Start all services"
            echo "  stop            - Stop all services"
            echo "  restart         - Restart all services"
            echo "  docker-help     - Show Docker commands"
            echo "  bash            - Run bash shell"
            echo "  bash -c 'cmd'   - Execute a bash command"
            exit 1
        fi
        ;;
esac

# End of script

