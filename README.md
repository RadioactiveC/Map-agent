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

3. Or set your DeepSeek API key (if you want to use DeepSeek via LangChain `ChatDeepSeek`):

```bash
export DEEPSEEK_API_KEY=ds-...
# Optional if you use a custom gateway
# export DEEPSEEK_API_BASE=https://api.deepseek.com
```

4. Run the agent:

```bash
# OpenAI
python main.py -q "生成广东省行政地图，高速公路用白色粗线表示，保存到 /workspace/out.png" -m gpt-4o-mini

# DeepSeek (model examples: deepseek-chat, deepseek-reasoner)
python main.py -q "使用 /workspace/data/guangdong_boundary.shp 作为底图，叠加 /workspace/data/highways.shp，线宽2，白色，保存到 /workspace/out.png" -m deepseek-chat
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

## Using your own basemap (Shapefile/GeoJSON/GPKG)

- Provide absolute paths to your data directly in你的自然语言指令中，代理会自动调用 `map_initial` 和 `map_add_layer` 去加载。例如：
  - `/workspace/data/basemap.shp`
  - `/workspace/data/roads.shp`
- 可一次性在初始化输入里传多个路径（JSON 数组或逗号分隔）。见 `map_initial` 工具说明。
- 受支持的矢量/栅格格式：`.shp`, `.geojson`/`.json`, `.gpkg`, `.tif`/`.tiff`。

## Notes

- The agent enforces the workflow: initialize first (map_initial) and save last (map_save).
- The plotting uses matplotlib's Agg backend (headless).
- If no LLM is configured, the program will raise a runtime error.
- DeepSeek integration requires `DEEPSEEK_API_KEY` and will be used automatically when `-m` 包含 `deepseek`。