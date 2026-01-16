
import pystac_client
from datetime import datetime

def test_search():
    catalog_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [10.0, 10.0],
            [10.1, 10.0],
            [10.1, 10.1],
            [10.0, 10.1],
            [10.0, 10.0]
        ]]
    }
    
    search = pystac_client.Client.open(catalog_url).search(
        collections=['sentinel-2-l2a'],
        intersects=geometry,
        datetime="2023-01-01/2023-01-31",
        max_items=1
    )
    
    items = list(search.get_items())
    if items:
        item = items[0]
        print(f"ID: {item.id}")
        print("Assets:")
        for key, asset in item.assets.items():
            print(f"  {key}: {asset.href} (Roles: {asset.roles})")
    else:
        print("No items found")

if __name__ == "__main__":
    test_search()
