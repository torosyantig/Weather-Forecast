
import subprocess

def check_installed_packages():
    required_packages = [
        "pandas",
        "scikit-learn",
        "joblib",
        "matplotlib",
        "streamlit",
        "plotly",
        "pydeck",
        "geopy"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"{package} is installed.")
        except ImportError:
            print(f"{package} is NOT installed.")

if __name__ == "__main__":
    check_installed_packages()
