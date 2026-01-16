
import pystac_client
import json

def test_props():
    catalog_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    search = pystac_client.Client.open(catalog_url).search(
        collections=['sentinel-2-l2a'],
        max_items=1
    )
    items = list(search.get_items())
    if items:
        print(json.dumps(items[0].properties, indent=2))
    else:
        print("No items found")

if __name__ == "__main__":
    test_props()
