import json
import requests
import pandas as pd
from io import StringIO
from tqdm import tqdm

def query_sdss(sql_query):
    url = "https://skyserver.sdss.org/dr16/SkyServerWS/SearchTools/SqlSearch"
    params = {
        "cmd": sql_query,
        "format": "csv"
    }

    response = requests.get(url, params=params)
    if response.status_code == 200 and "error" not in response.text.lower():
        try:
            df = pd.read_csv(StringIO(response.text))
            return df.head(5), None
        except Exception as e:
            return None, f"CSV parsing error: {e}"
    else:
        return None, f"HTTP error {response.status_code} or SQL syntax issue"


def main():
    with open("data/generated/phi2-method1-v3.json") as f:
        data = json.load(f)

    results = []
    for i, item in enumerate(tqdm(data, desc="Validating SQL queries")):
        sql = item["output"]
        sql_mod = sql.replace("SELECT ", "SELECT TOP 10 ", 1)
        table, error = query_sdss(sql_mod)
        results.append({
            "index": i,
            "sql": sql_mod,
            "valid": error is None,
            "error": error,
            "row_count": len(table) if table is not None else 0
        })

    df = pd.DataFrame(results)
    df.to_csv("data/generated/phi2-method1-v3.csv", index=False)
    print("Validation complete. Output saved to data/generated/phi2-method1-v3.csv")

if __name__ == "__main__":
    main()