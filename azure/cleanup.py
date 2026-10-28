"""Delete the online endpoint. It bills per hour while it exists, even when idle."""
from common import ENDPOINT_NAME, get_client

if __name__ == "__main__":
    get_client().online_endpoints.begin_delete(name=ENDPOINT_NAME).result()
    print(f"Deleted endpoint {ENDPOINT_NAME}")
