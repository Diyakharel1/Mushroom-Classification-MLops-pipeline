"""
Model serving utilities for the mushroom classification project.
This module provides utilities for Airflow user management and authentication.
"""

import subprocess
import os
import sys
import time


def check_airflow_version():
    """Check Airflow version and return version info"""
    try:
        result = subprocess.run(["airflow", "version"], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Airflow version: {version}")

            # Check if this is Airflow 3.x
            if "3." in version:
                return "3.x"
            else:
                return "2.x"
        else:
            print("❌ Airflow not found")
            return None
    except Exception as e:
        print(f"❌ Error checking Airflow: {e}")
        return None


def stop_all_airflow():
    """Stop all running Airflow processes"""
    try:
        print("🛑 Stopping all Airflow processes...")
        subprocess.run(["pkill", "-f", "airflow"], capture_output=True)
        time.sleep(2)
        print("✅ All Airflow processes stopped")
    except Exception as e:
        print(f"⚠️ Error stopping processes: {e}")


def fix_airflow_v3():
    """Fix Airflow 3.x using standalone mode with proper credentials"""
    print("🔧 Fixing Airflow 3.x with standalone mode")
    print("=" * 45)

    # Set environment
    airflow_home = os.path.join(os.getcwd(), "airflow")
    os.environ["AIRFLOW_HOME"] = airflow_home
    os.environ["AIRFLOW__CORE__LOAD_EXAMPLES"] = "False"
    os.environ["AIRFLOW__WEBSERVER__EXPOSE_CONFIG"] = "True"

    # Set default admin credentials for standalone mode
    os.environ["_AIRFLOW_WWW_USER_USERNAME"] = "admin"
    os.environ["_AIRFLOW_WWW_USER_PASSWORD"] = "admin"

    print(f"🔧 Using Airflow Home: {airflow_home}")

    # Create directories
    os.makedirs(f"{airflow_home}/dags", exist_ok=True)
    os.makedirs(f"{airflow_home}/logs", exist_ok=True)

    # Stop any existing processes
    stop_all_airflow()

    # Step 1: Initialize database
    print("1️⃣ Initializing Airflow database...")
    result = subprocess.run(["airflow", "db", "init"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Database initialized")
    else:
        print(f"⚠️ Database init: {result.stderr}")

    # Step 2: Start standalone mode
    print("2️⃣ Starting Airflow in standalone mode...")
    print("   This creates admin/admin credentials automatically")

    try:
        # Start standalone in background and redirect output to file
        log_file = os.path.join(airflow_home, "standalone.log")
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                [
                    "airflow",
                    "standalone",
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
            )

        print(f"⏳ Starting Airflow... (logs: {log_file})")
        print("   Waiting 30 seconds for services to start...")

        # Wait for startup
        time.sleep(30)

        # Check if process is still running
        poll = process.poll()

        if poll is None:  # Process is still running
            print("✅ Airflow standalone mode is running!")

            # Try to access the web interface
            try:
                import requests

                response = requests.get("http://localhost:8080", timeout=10)
                if response.status_code == 200:
                    print("✅ Web interface is accessible!")
                    print("")
                    print("🔗 URL: http://localhost:8080")
                    print("👤 Username: admin")
                    print("🔑 Password: admin")
                    print("")
                    print("📋 To view logs: tail -f " + log_file)
                    print("🛑 To stop: pkill -f 'airflow standalone'")
                    return True
                else:
                    print(f"⚠️ Web interface returned status: {response.status_code}")
            except Exception as e:
                print(f"⚠️ Cannot access web interface yet: {e}")
                print("   Services may still be starting up...")
                print("   Try accessing http://localhost:8080 in a minute")

            return True
        else:
            print("❌ Standalone mode failed to start")
            # Show last few lines of log
            try:
                with open(log_file, "r") as f:
                    lines = f.readlines()
                    print("Last few log lines:")
                    for line in lines[-10:]:
                        print(f"   {line.strip()}")
            except:
                pass
            return False

    except Exception as e:
        print(f"❌ Error starting standalone mode: {e}")
        return False


def reset_airflow_user():
    """Reset Airflow admin user credentials"""
    try:
        # Set Airflow environment
        airflow_home = os.environ.get(
            "AIRFLOW_HOME", os.path.join(os.getcwd(), "airflow")
        )
        os.environ["AIRFLOW_HOME"] = airflow_home

        print(f"🔧 Using Airflow Home: {airflow_home}")

        # Delete existing user if exists
        print("🗑️ Removing existing admin user...")
        subprocess.run(
            ["airflow", "users", "delete", "--username", "admin"], capture_output=True
        )

        # Create new admin user
        print("👤 Creating new admin user...")
        result = subprocess.run(
            [
                "airflow",
                "users",
                "create",
                "--username",
                "admin",
                "--firstname",
                "Admin",
                "--lastname",
                "User",
                "--role",
                "Admin",
                "--email",
                "admin@example.com",
                "--password",
                "admin",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ Airflow admin user created successfully")
            print("   Username: admin")
            print("   Password: admin")
            print("   URL: http://localhost:8080")
            return True
        else:
            print(f"❌ Failed to create user: {result.stderr}")
            print(
                "💡 Make sure Airflow is properly installed and database is initialized"
            )
            return False

    except Exception as e:
        print(f"❌ Error resetting user: {e}")
        print("💡 Try running: airflow db init")
        return False


def check_airflow_status():
    """Check if Airflow services are running"""
    try:
        # Check if airflow command is available
        result = subprocess.run(["airflow", "version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Airflow version: {result.stdout.strip()}")
            return True
        else:
            print("❌ Airflow not found or not working")
            return False
    except Exception as e:
        print(f"❌ Error checking Airflow: {e}")
        return False


def start_airflow_services():
    """Start Airflow webserver and scheduler"""
    try:
        airflow_home = os.environ.get(
            "AIRFLOW_HOME", os.path.join(os.getcwd(), "airflow")
        )
        os.environ["AIRFLOW_HOME"] = airflow_home

        print("🚀 Starting Airflow services...")

        # Start webserver
        print("🌐 Starting Airflow webserver...")
        webserver = subprocess.Popen(
            ["airflow", "webserver", "--port", "8080", "--daemon"]
        )

        # Start scheduler
        print("📅 Starting Airflow scheduler...")
        scheduler = subprocess.Popen(["airflow", "scheduler", "--daemon"])

        print("✅ Airflow services started!")
        print("   Access: http://localhost:8080")
        print("   Username: admin")
        print("   Password: admin")

        return True

    except Exception as e:
        print(f"❌ Error starting Airflow: {e}")
        return False


def check_and_start_webserver():
    """Check if webserver is running and start if needed"""
    try:
        import psutil
        import time

        # Check if webserver is already running
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if "airflow" in proc.info["name"] and "webserver" in " ".join(
                    proc.info["cmdline"]
                ):
                    print(
                        f"✅ Airflow webserver already running (PID: {proc.info['pid']})"
                    )
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Start webserver if not running
        print("🌐 Starting Airflow webserver...")
        airflow_home = os.environ.get(
            "AIRFLOW_HOME", os.path.join(os.getcwd(), "airflow")
        )
        os.environ["AIRFLOW_HOME"] = airflow_home

        # Start webserver in background
        subprocess.Popen(["airflow", "webserver", "--port", "8080", "--daemon"])

        # Wait and check if it started
        time.sleep(5)

        # Check again if webserver is now running
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if "airflow" in proc.info["name"] and "webserver" in " ".join(
                    proc.info["cmdline"]
                ):
                    print(f"✅ Airflow webserver started (PID: {proc.info['pid']})")
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        print("❌ Failed to start webserver")
        return False

    except ImportError:
        # Fallback without psutil
        print("🌐 Starting Airflow webserver (fallback method)...")
        subprocess.Popen(["airflow", "webserver", "--port", "8080", "--daemon"])
        return True

    except Exception as e:
        print(f"❌ Error checking/starting webserver: {e}")
        return False


def fix_airflow_completely():
    """Complete Airflow fix - detect version and use appropriate method"""
    print("🔧 Complete Airflow Fix")
    print("=" * 30)

    version = check_airflow_version()

    if version == "3.x":
        return fix_airflow_v3()
    elif version == "2.x":
        print("⚠️ Airflow 2.x detected - using legacy method")
        # Set environment
        airflow_home = os.path.join(os.getcwd(), "airflow")
        os.environ["AIRFLOW_HOME"] = airflow_home

        # Step 1: Initialize database
        print("1️⃣ Initializing Airflow database...")
        subprocess.run(["airflow", "db", "init"], capture_output=True)

        # Step 2: Reset user
        print("2️⃣ Resetting admin user...")
        reset_airflow_user()

        # Step 3: Start webserver
        print("3️⃣ Starting webserver...")
        check_and_start_webserver()

        # Step 4: Final status check
        print("4️⃣ Final status check...")
        time.sleep(3)

        try:
            import requests

            response = requests.get("http://localhost:8080", timeout=5)
            if response.status_code == 200:
                print("✅ Airflow webserver is accessible!")
                print("🔗 URL: http://localhost:8080")
                print("👤 Username: admin")
                print("🔑 Password: admin")
                return True
            else:
                print(f"⚠️ Webserver responded with status: {response.status_code}")
        except Exception as e:
            print(f"❌ Cannot access webserver: {e}")
            print("💡 Try manually: airflow webserver --port 8080")

        return False
    else:
        print("❌ Could not detect Airflow version")
        return False


if __name__ == "__main__":
    print("🔧 Airflow Complete Fix Utility")
    print("=" * 40)

    success = fix_airflow_completely()
    if success:
        print("\n🎉 Airflow should be ready!")
    else:
        print("\n❌ Manual intervention needed")
