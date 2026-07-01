import streamlit as st
import folium
import rasterio
import numpy as np
import os
import pandas as pd
import plotly.express as px
from streamlit_folium import st_folium
from pyproj import Transformer
from folium.raster_layers import ImageOverlay
from io import BytesIO
import base64
import rasterio.transform
from pathlib import Path
from PIL import Image
from rasterio.enums import Resampling
from rasterio.features import shapes
import geopandas as gpd

# --- PAGE CONFIG ---
st.set_page_config(
    layout="wide", 
    page_title="Groundwater Potential Zone mapping of Medchal-Malkajgiri District",
)

st.title("💧 Groundwater Potential Zone Mapping of Medchal-Malkajgiri District")

# =====================================================================
# 🎛️ MASTER LAYER MAPPINGS
# =====================================================================
LAYER_MAPPINGS = {
    "Groundwater Potential": {
        "classes": {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"},
        "colors": {1: (255, 0, 0, 255), 2: (255, 165, 0, 255), 3: (255, 255, 0, 255), 4: (76, 175, 80, 255), 5: (0, 100, 0, 255)}
    },
    "Geomorphology": {
        "classes": {1: "Low Dissected Hills and Valleys", 2: "Waterbody - River", 3: "Quarry and Mine Dump", 4: "Waterbodies-Other", 5: "Moderately Dissected Hills and Valleys", 6: "Flood Plain", 7: "Pediment Pediplain Complex"}, 
        "colors": {1: (255, 255, 0, 255), 2: (0, 191, 255, 255), 3: (128, 0, 128, 255), 4: (0, 0, 255, 255), 5: (34, 139, 34, 255), 6: (144, 238, 144, 255), 7: (255, 140, 0, 255)}
    },
    "Soil": {
        "classes": {0: "Acrisols", 6: "Cambisols", 11: "Fluvisols", 16: "Leptosols", 18: "Luvisols", 29: "Vertisols"},
        "colors": {0: (144, 238, 144), 6: (245, 245, 220), 11: (30, 144, 255), 16: (255, 69, 0), 18: (139, 69, 19), 29: (107, 142, 35)}
    },
    "LULC": {
        "classes": {1: "Urban", 9: "Water Bodies", 22: "Vegetation", 30: "Barren Land", 42: "Agriculture"},
        "colors": {1: (255, 0, 0, 255), 9: (0, 0, 255, 255), 22: (34, 139, 34, 255), 30: (222, 184, 135, 255), 42: (144, 238, 144, 255)}
    },
    "Geology": {
        "classes": {1: "Basic Rocks", 2: "Granitic Rocks", 3: "Quartz Vein", 4: "Hornblende Granites", 5: "Gneissic Rocks"},
        "colors": {1: (50, 205, 50, 255), 2: (255, 192, 203, 255), 3: (240, 240, 240, 255), 4: (128, 0, 128, 255), 5: (220, 20, 60, 255)}
    },
    "Rainfall": {
        "classes": {1: "Very Low Rainfall", 2: "Low Rainfall", 3: "Moderate Rainfall", 4: "High Rainfall", 5: "Very High Rainfall"},
        "colors": {1: (255, 165, 0, 255), 2: (255, 255, 0, 255), 3: (0, 128, 0, 255), 4: (0, 255, 255, 255), 5: (0, 0, 139, 255)}
    },
    "Lineament Density": {
        "classes": {1: "Very Low Density", 2: "Low Density", 3: "Moderate Density", 4: "High Density", 5: "Very High Density"},
        "colors": {1: (34, 139, 34, 255), 2: (144, 238, 144, 255), 3: (255, 255, 0, 255), 4: (255, 140, 0, 255), 5: (255, 0, 0, 255)}
    },
    "Drainage Density": {
        "classes": {1: "Very Low Density", 2: "Low Density", 3: "Moderate Density", 4: "High Density", 5: "Very High Density"},
        "colors": {1: (0, 0, 139, 255), 2: (0, 191, 255, 255), 3: (255, 255, 0, 255), 4: (255, 165, 0, 255), 5: (255, 0, 0, 255)}
    },
    "Slope": {
        "classes": {1: "Very High", 2: "High", 3: "Moderate", 4: "Low", 5: "Very Low"},
        "colors": {1: (34, 139, 34, 255), 2: (152, 251, 152, 255), 3: (255, 255, 0, 255), 4: (255, 165, 0, 255), 5: (255, 0, 0, 255)}
    }
}

# ---------------- FILE PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR.parent.parent / "GWPZM OUTPUT"

MAPS_REGISTRY = {
    "Rainfall Map": str(OUTPUT_DIR / "outputfmt/format/Rainfall map.png"),
    "Soil Map": str(OUTPUT_DIR / "outputfmt/format/soil map.png"),
    "Slope Map": str(OUTPUT_DIR / "outputfmt/format/Slope map.png"),
    "LULC Map": str(OUTPUT_DIR / "outputfmt/format/LULC MAP.png"), 
    "Lineament Density Map": str(OUTPUT_DIR / "outputfmt/format/Lineament density.png"),
    "Geomorphology Map": str(OUTPUT_DIR / "outputfmt/format/Geomorphology.png"),
    "Geology Map": str(OUTPUT_DIR / "outputfmt/format/Geology map.png"),
    "Drainage Density Map": str(OUTPUT_DIR / "outputfmt/format/Drainage density.png")
}

gw_tif_path = str(BASE_DIR / "tif1.tif")

TIFF_LAYERS_REGISTRY = {
    "Rainfall": str(OUTPUT_DIR / "outputfmt/format/tiffs/Rainfall1.tif"),
    "Geomorphology": str(OUTPUT_DIR / "outputfmt/format/tiffs/Geomorphology1.tif"),
    "Geology": str(OUTPUT_DIR / "outputfmt/format/tiffs/Geology1.tif"),
    "Lineament Density": str(OUTPUT_DIR / "outputfmt/format/tiffs/Lineament Density1.tif"),
    "Drainage Density": str(OUTPUT_DIR / "outputfmt/format/tiffs/Drainage Density1.tif"),
    "Slope": str(OUTPUT_DIR / "outputfmt/format/tiffs/Slope1.tif"),
    "Soil": str(OUTPUT_DIR / "outputfmt/format/tiffs/soil.tif"),
    "LULC": str(OUTPUT_DIR / "outputfmt/format/tiffs/LULC1.tif")
}

# ---------------- SHAPEFILE PATH ----------------
village_shp_path = r"C:\Users\Chatl\OneDrive\Pictures\Documents\Pictures\Desktop\web application\GWPZM WEB\village boundary of medchal\village bundary.shp"

# ---------------- CORE PROCESSING FUNCTIONS ----------------
@st.cache_data
def load_raster_data(filepath):
    if not os.path.exists(filepath): return None, None, None, None, None, None
    try:
        with rasterio.open(filepath) as src:
            return src.read(1), src.bounds, src.transform, src.width, src.height, src.crs.to_string() if src.crs else "EPSG:4326"
    except:
        return None, None, None, None, None, None

data, bounds, raster_transform, width, height, crs_string = load_raster_data(gw_tif_path)
ALL_LAYERS_CACHE = {name: {"data": d[0], "bounds": d[1], "transform": d[2], "crs": d[5], "path": path} 
                    for name, path in TIFF_LAYERS_REGISTRY.items() if (d := load_raster_data(path))[0] is not None}

@st.cache_data
def create_hover_polygons(filepath, classes_mapping, scale_factor=0.05):
    if not os.path.exists(filepath): return gpd.GeoDataFrame()
    with rasterio.open(filepath) as src:
        new_w, new_h = int(src.width * scale_factor), int(src.height * scale_factor)
        r_data = src.read(1, out_shape=(new_h, new_w), resampling=Resampling.nearest)
        transform = rasterio.transform.from_bounds(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top, new_w, new_h)
        shape_gen = shapes(r_data.astype(np.int32), mask=r_data>0, transform=transform)
        gdf = gpd.GeoDataFrame.from_features([{'properties': {'raster_val': int(v)}, 'geometry': s} for s, v in shape_gen], crs=src.crs)
        gdf['Class_Name'] = gdf['raster_val'].map(classes_mapping).fillna("Unknown Class")
        return gdf.to_crs(epsg=4326).simplify(0.01)

hover_gdf_gw = create_hover_polygons(gw_tif_path, LAYER_MAPPINGS["Groundwater Potential"]["classes"])

@st.cache_data
def load_shapefile(filepath):
    if not os.path.exists(filepath): return gpd.GeoDataFrame()
    try:
        gdf = gpd.read_file(filepath)
        return gdf.to_crs(epsg=4326) # Map requires standard lat/long
    except:
        return gpd.GeoDataFrame()

@st.cache_data
def calculate_area_statistics(data_array, layer_config, transform):
    unique, counts = np.unique(data_array, return_counts=True)
    count_dict = dict(zip(unique, counts))
    res_x, res_y = abs(transform[0]), abs(transform[4])
    pixel_area = (res_x * res_y) / 1_000_000.0 if res_x > 0.1 else ((res_x * 111320 * np.cos(np.radians(17.5))) * (res_y * 111000)) / 1_000_000.0
    total = sum(count_dict.get(k, 0) for k, name in layer_config.get("classes", {}).items() if name != "Unknown")
    
    stats = [{"Zone": name, "Pixels": count_dict.get(key, 0), "Percentage": (count_dict.get(key, 0) / total * 100) if total > 0 else 0, 
              "Area (Sq.Km)": count_dict.get(key, 0) * pixel_area, "Color": '#{:02x}{:02x}{:02x}'.format(*layer_config.get("colors", {}).get(key, (128, 128, 128))[:3])}
             for key, name in layer_config.get("classes", {}).items() if name != "Unknown"]
    return pd.DataFrame(stats)

@st.cache_data
def generate_overlay_image_fast(data_array, colors_config):
    if data_array is None: return None
    rgba = np.zeros((*data_array.shape, 4), dtype=np.uint8)
    for val, color in colors_config.items(): rgba[data_array == val] = color if len(color) == 4 else (*color, 255)
    buf = BytesIO()
    Image.fromarray(rgba, 'RGBA').save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


# =====================================================================
# 🖥️ STABLE SPLIT-SCREEN DASHBOARD LAYOUT
# =====================================================================
left_panel, right_panel = st.columns([1, 2.5]) # Fixed column ratios

# --- LEFT PANEL: ALL CONTROLS ---
with left_panel:
    st.header("📊 Map Controls")
    details_placeholder = st.container() # Reserving space for clicks
    
    st.markdown("---")
    st.subheader("📍 Vector Overlays")
    show_villages = st.checkbox("Show Village Boundaries", value=False)
    
    st.markdown("---")
    st.subheader("🗺️ Raster Toggles")
    active_layers = {}
    checked_layers_list = []

    if st.checkbox("Show Groundwater Potential Zones", value=True):
        checked_layers_list.append("Groundwater Potential")
        active_layers["Groundwater Potential"] = {"url": generate_overlay_image_fast(data, LAYER_MAPPINGS["Groundwater Potential"]["colors"]), "bounds": bounds, "crs": crs_string, "hover": hover_gdf_gw}

    for name in TIFF_LAYERS_REGISTRY.keys():
        if st.checkbox(f"Show {name}"):
            checked_layers_list.append(name)
            l_info = ALL_LAYERS_CACHE.get(name)
            if l_info:
                active_layers[name] = {"url": generate_overlay_image_fast(l_info["data"], LAYER_MAPPINGS.get(name, {}).get("colors", {})), 
                                       "bounds": l_info["bounds"], "crs": l_info["crs"], "hover": create_hover_polygons(l_info["path"], LAYER_MAPPINGS.get(name, {}).get("classes", {}))}

    st.markdown("---")
    global_opacity = st.slider("Global Raster Opacity", 0.0, 1.0, 0.8, 0.1)

# --- RIGHT PANEL: TABS AND MAP ---
with right_panel:
    tab1, tab2, tab3 = st.tabs(["💧 Interactive Portal", "🖼️ Maps Gallery", "📄 Data Information"])
    
    with tab1:
        # CENTER MAP
        if bounds and crs_string:
            trans = Transformer.from_crs(crs_string, "EPSG:4326", always_xy=True)
            lon1, lat1 = trans.transform(bounds.left, bounds.bottom)
            lon2, lat2 = trans.transform(bounds.right, bounds.top)
            center_lat, center_lon = (lat1 + lat2) / 2, (lon1 + lon2) / 2
        else:
            center_lat, center_lon = 17.5, 78.5 # Fallback
            
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", attr="Google Satellite")

        # 1. ADD RASTERS
        for name, info in active_layers.items():
            trans_layer = Transformer.from_crs(info["crs"], "EPSG:4326", always_xy=True)
            l1, b1 = trans_layer.transform(info["bounds"].left, info["bounds"].bottom)
            l2, b2 = trans_layer.transform(info["bounds"].right, info["bounds"].top)
            ImageOverlay(info["url"], bounds=[[b1, l1], [b2, l2]], opacity=global_opacity).add_to(m)
            if not info["hover"].empty:
                folium.GeoJson(info["hover"], tooltip=folium.GeoJsonTooltip(fields=['Class_Name'], aliases=[f'📍 {name}:'])).add_to(m)

        # 2. ADD SHAPEFILE (TRANSPARENT WITH BORDER)
        if show_villages:
            village_gdf = load_shapefile(village_shp_path)
            if not village_gdf.empty:
                folium.GeoJson(
                    village_gdf,
                    name="Village Boundaries",
                    style_function=lambda feature: {
                        'color': 'black',        # Border color
                        'weight': 1.5,           # Border thickness
                        'fillColor': '#000000',  # Irrelevant but required by folium
                        'fillOpacity': 0.0       # 0.0 = completely transparent inside
                    }
                ).add_to(m)
            else:
                st.error("⚠️ Village boundary shapefile not found at specified path.")

        # RENDER MAP
        map_data = st_folium(m, height=550, use_container_width=True, returned_objects=["last_clicked"])

        # PROCESS CLICKS
        with details_placeholder:
            st.subheader("📍 Clicked Location Details")
            if map_data and map_data.get("last_clicked"):
                lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
                st.markdown(f"**Lat:** {lat:.4f} | **Lon:** {lon:.4f}")
                
                def get_pixel_value(t_data, t_trans, t_crs, target_lat, target_lon):
                    try:
                        x, y = Transformer.from_crs("EPSG:4326", t_crs, always_xy=True).transform(target_lon, target_lat)
                        row, col = rasterio.transform.rowcol(t_trans, x, y)
                        if 0 <= row < t_data.shape[0] and 0 <= col < t_data.shape[1]: return int(t_data[row, col])
                    except: pass
                    return None

                for active_name in active_layers.keys():
                    if active_name == "Groundwater Potential":
                        val = get_pixel_value(data, raster_transform, crs_string, lat, lon)
                    else:
                        info = ALL_LAYERS_CACHE.get(active_name)
                        val = get_pixel_value(info["data"], info["transform"], info["crs"], lat, lon) if info else None

                    if val is not None and val in LAYER_MAPPINGS.get(active_name, {}).get("classes", {}):
                        text = LAYER_MAPPINGS[active_name]["classes"][val]
                        bg_rgba = LAYER_MAPPINGS[active_name]["colors"][val]
                        st.markdown(f'<div style="background-color:#{:02x}{:02x}{:02x}; padding:5px; border-radius:5px; margin-bottom:5px; color:{"black" if ((bg_rgba[0]*299 + bg_rgba[1]*587 + bg_rgba[2]*114)/1000) > 125 else "white"};"><b>{active_name}:</b> {text}</div>', unsafe_allow_html=True)
            else:
                st.info("Click the map to identify layer zones.")

        # STATISTICS
        st.markdown("---")
        st.subheader("📊 Spatial Attribute Statistics")
        if checked_layers_list:
            selected_layer = checked_layers_list[-1]
            stat_data = data if selected_layer == "Groundwater Potential" else ALL_LAYERS_CACHE[selected_layer]["data"]
            stat_trans = raster_transform if selected_layer == "Groundwater Potential" else ALL_LAYERS_CACHE[selected_layer]["transform"]
            
            df_stats = calculate_area_statistics(stat_data, LAYER_MAPPINGS.get(selected_layer, {}), stat_trans)
            if not df_stats.empty and df_stats['Pixels'].sum() > 0:
                c1, c2 = st.columns([1, 1])
                with c1:
                    fig = px.pie(df_stats, values='Percentage', names='Zone', color='Zone', color_discrete_map={r['Zone']: r['Color'] for _, r in df_stats.iterrows()}, hole=0.4)
                    fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#000000', width=1)))
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    st.write(f"**Area Coverage: {selected_layer}**")
                    st.dataframe(df_stats[['Zone', 'Area (Sq.Km)', 'Percentage']].style.format({'Area (Sq.Km)': '{:.2f} km²', 'Percentage': '{:.2f}%'}), hide_index=True)

    with tab2:
        st.subheader("🖼️ Static Maps Inventory Gallery")
        gallery_maps = list(MAPS_REGISTRY.items())
        for i in range(0, len(gallery_maps), 2):
            cols = st.columns(2)
            for j, (name, path) in enumerate(gallery_maps[i:i+2]):
                if os.path.exists(path): cols[j].image(path, caption=name, use_container_width=True)
                else: cols[j].error(f"Missing: `{name}`")

    with tab3:
        st.header("📄 Data Sources & Methodology")
        st.markdown("""
        This WebGIS portal maps Groundwater Potential Zones using a Multi-Criteria Decision Making (MCDM) approach. 
        The primary datasets were sourced based on their spatial resolution, temporal relevance, and scientific authenticity.

        ### 🗂️ Thematic Layers & Sources
        | Dataset | Source | Type / Resolution | Primary Purpose |
        | :--- | :--- | :--- | :--- |
        | **DEM** | Bhoonidhi Geoportal | 30m Raster | Terrain, slope, and drainage analysis |
        | **Surface Cover** | Bhoonidhi (LISS-IV) | Raster | LULC Classification,recharge assessment |
        | **Rainfall** | IMD | Annual Data | Groundwater recharge estimation |
        | **Soil** | SoilGrids | Raster | Soil texture and infiltration capacity |
        | **Geology** | NGDR | Vector | Lithological and aquifer characterization |
        | **Geomorphology** | NGDR | Vector | Landform and structural analysis |
        | **Lineaments** | Bhuvan WMS | Vector | Fault and fracture density mapping |
        | **Drainage** | Derived from DEM | Vector | Runoff and stream network density |
        """)
        
        st.subheader("⚙️ Analytical Processing (AHP)")
        st.markdown("""
        The individual thematic layers were processed in a GIS environment. We utilized **Saaty’s Analytical Hierarchy Process (AHP)** to calculate the normalized weights for each layer based on their relative importance to groundwater occurrence. 
        
        The final Groundwater Potential Index (GWPI) was calculated using a Weighted Overlay Analysis, classifying the region into five distinct zones ranging from **Very Low** to **Very High**.
        """)
        st.latex(r"GWPI = \sum_{i=1}^{n} (W_i \times R_i)")
