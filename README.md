# MapGPT (Agent)

A Thought-Action-Observation style agent for map generation using vector and raster data.

## Quick Start

1. Install dependencies (requires GEOS/GDAL stack for GeoPandas/Rasterio):

```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key (if you want to use OpenAI models):

```bash
export OPENAI_API_KEY=sk-...  # or use your key provider
```

3. Run the agent:

```bash
python main.py -q "生成广东省行政地图，高速公路用白色粗线表示，保存到 /workspace/out.png" -m gpt-4o-mini
```

Use `--list-tools` to see available tools:

```bash
python main.py --list-tools
```

## Tools

Tools are defined in `mapgpt/tools.py`. They include:
- map_initial: initialize the map canvas
- modify_line_width / modify_line_color
- map_add_layer: add vector (.shp/.geojson/.gpkg) or raster (.tif)
- map_set_title, map_set_background_color
- map_save: save the map image

## Notes

- The agent enforces the workflow: initialize first (map_initial) and save last (map_save).
- The plotting uses matplotlib's Agg backend (headless).
- If no LLM is configured, the program will raise a runtime error.