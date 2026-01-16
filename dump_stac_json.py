
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
        props = items[0].properties
        with open("stac_props_clean.json", "w", encoding="utf-8") as f:
            json.dump({
                "collection": items[0].collection_id,
                "properties": props
            }, f, indent=2)
        print("Wrote to stac_props_clean.json")
    else:
        print("No items found")

if __name__ == "__main__":
    test_props()
