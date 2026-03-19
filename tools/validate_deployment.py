import importlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
import py_compile

try:
    import tomllib
except Exception:  # pragma: no cover
    tomllib = None


CORE_IMPORTS = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "numpy": "numpy",
    "plotly": "plotly",
    "scipy": "scipy",
    "rapidfuzz": "rapidfuzz",
    "openpyxl": "openpyxl",
    "xlsxwriter": "xlsxwriter",
    "reportlab": "reportlab",
}

OPTIONAL_IMPORTS = {
    "ydata-profiling": "ydata_profiling",
    "featuretools": "featuretools",
    "kaleido": "kaleido",
    "boto3": "boto3",
    "tableauserverclient": "tableauserverclient",
    "tableauhyperapi": "tableauhyperapi",
    "msal": "msal",
    "requests": "requests",
}


def load_secrets():
    path = Path(".streamlit/secrets.toml")
    if not path.exists() or tomllib is None:
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if "general" in data and isinstance(data["general"], dict):
        merged = dict(data)
        general = merged.pop("general")
        merged.update(general)
        return merged
    return data


def get_secret_value(key: str, secrets: dict):
    if key in secrets and secrets[key]:
        return secrets[key]
    return os.environ.get(key)


def check_imports(label: str, mapping: dict):
    missing = {}
    for pkg, mod in mapping.items():
        try:
            importlib.import_module(mod)
        except Exception as e:
            missing[pkg] = f"{type(e).__name__}: {e}"
    if missing:
        print(f"{label} missing:")
        for pkg, err in missing.items():
            print(f"- {pkg}: {err}")
    else:
        print(f"{label} OK")
    return missing


def check_py_compile(paths):
    errors = []
    for path in paths:
        try:
            py_compile.compile(path, doraise=True)
        except Exception as e:
            errors.append((path, str(e)))
    if errors:
        for p, err in errors:
            print(f"Compile error: {p}: {err}")
    else:
        print("Python compile OK")
    return errors


def wait_for_port(host: str, port: int, timeout_s: int = 20) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except Exception:
            time.sleep(0.5)
    return False


def check_streamlit_launch():
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.headless",
        "true",
        "--server.port",
        "8502",
        "--server.address",
        "127.0.0.1",
    ]
    env = dict(os.environ)
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        ok = wait_for_port("127.0.0.1", 8502, timeout_s=20)
        print("Streamlit launch OK" if ok else "Streamlit launch FAILED")
        return ok
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


def check_secrets():
    secrets = load_secrets()
    groups = {
        "S3": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"],
        "Tableau": ["TABLEAU_SERVER", "TABLEAU_SITE", "TABLEAU_PROJECT", "TABLEAU_TOKEN_NAME", "TABLEAU_TOKEN_VALUE"],
        "PowerBI": ["PBI_TENANT", "PBI_CLIENT_ID", "PBI_CLIENT_SECRET", "PBI_GROUP_ID"],
    }
    for group, keys in groups.items():
        present = all(get_secret_value(k, secrets) for k in keys)
        status = "configured" if present else "missing"
        print(f"Secrets {group}: {status}")


def main():
    missing_core = check_imports("Core dependencies", CORE_IMPORTS)
    missing_optional = check_imports("Optional dependencies", OPTIONAL_IMPORTS)

    compile_errors = check_py_compile(
        [
            "app.py",
            "reporting.py",
            "ai_insights.py",
            "data_loading.py",
            "cleaning_tools.py",
            "feature_engineering.py",
            "visualization.py",
            "powerbi_publisher.py",
            "tableau_publisher.py",
            "modules/background_tasks.py",
        ]
    )

    check_secrets()
    streamlit_ok = check_streamlit_launch()

    if missing_core or compile_errors or not streamlit_ok:
        print("Deployment validation FAILED")
        sys.exit(1)

    if missing_optional:
        print("Deployment validation OK (some optional deps missing)")
        sys.exit(0)

    print("Deployment validation OK")


if __name__ == "__main__":
    main()
