import streamlit as st
import folium
import rasterio
import numpy as np
import os
from streamlit_folium import st_folium
from pyproj import Transformer
from folium.raster_layers import ImageOverlay
from io import BytesIO
import base64
import rasterio.transform

# --- IMPORTS FOR HIGH-SPEED PROCESSING ---
from PIL import Image
from rasterio.enums import Resampling
from rasterio.features import shapes
import geopandas as gpd

# --- IMPORTS FOR DASHBOARD ---
import plotly.express as px
import pandas as pd

hide_github_badge = """
<style>
    /* This targets the GitHub/Fork badge specifically */
    .viewerBadge_container__1QSob,
    .viewerBadge_link__1S137,
    div[class^="viewerBadge_container"],
    div[class*="viewerBadge"] {
        display: none !important;
           
# 1. Add some blank space at the very top of the sidebar to push content down
st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)

# 2. Your existing sidebar code follows...
st.sidebar.write("📍 **Clicked Location Details**")
# ... rest of your sidebar widgets ...
    }
</style>
"""

st.markdown(hide_github_badge, unsafe_allow_html=True)

# ---> Your app content follows below <---

st.set_page_config(layout="wide", page_title="Groundwater Potential Zone mapping of Medchal-Malkajgiri District")

# ---------------- TITLE ----------------
st.title("💧 Groundwater Potential Zone Mapping of Medchal-Malkajgiri District")

# =====================================================================
# 🎛️ MASTER LAYER MAPPINGS
# =====================================================================
LAYER_MAPPINGS = {
    "Groundwater Potential": {
        "classes": {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"},
        "colors": {
            1: (255, 0, 0, 255),      # Pure Red
            2: (255, 165, 0, 255),    # Orange
            3: (255, 255, 0, 255),    # Bright Yellow
            4: (76, 175, 80, 255),    # Medium Green
            5: (0, 100, 0, 255)       # Dark Green
        }
    },
    "Geomorphology": {
        "classes": {
            1: "Low Dissected Hills and Valleys", 
            2: "Waterbody - River", 
            3: "Quarry and Mine Dump", 
            4: "Waterbodies-Other", 
            5: "Moderately Dissected Hills and Valleys",
            6: "Flood Plain", 
            7: "Pediment Pediplain Complex"
        }, 
        "colors": {
            1: (255, 255, 0, 255),      # Yellow
            2: (0, 191, 255, 255),      # Light Blue
            3: (128, 0, 128, 255),      # Purple
            4: (0, 0, 255, 255),        # Blue
            5: (34, 139, 34, 255),      # Dark Green
            6: (144, 238, 144, 255),    # Light Green
            7: (255, 140, 0, 255)       # Orange
        }
    },
   "Soil": {
        "classes": {
            0: "Acrisols",
            6: "Cambisols",
            11: "Fluvisols",
            16: "Leptosols",
            18: "Luvisols",
            29: "Vertisols"
        },
        "colors": {
            0: (144, 238, 144),    
            6: (245, 245, 220),    
            11: (30, 144, 255),    
            16: (255, 69, 0),      
            18: (139, 69, 19),     
            29: (107, 142, 35)     
        }
    },
    "LULC": {
        "classes": {
            1: "Urban", 
            9: "Water Bodies", 
            22: "Vegetation", 
            30: "Barren Land", 
            42: "Agriculture",
        },
        "colors": {
            1: (255, 0, 0, 255),       # Red
            9: (0, 0, 255, 255),       # Blue
            22: (34, 139, 34, 255),    # Dark Green
            30: (222, 184, 135, 255),  # Tan
            42: (144, 238, 144, 255),  # Light Green
        }
    },
    "Geology": {
        "classes": {
            1: "Basic Rocks", 
            2: "Granitic Rocks", 
            3: "Quartz Vein", 
            4: "Hornblende Granites", 
            5: "Gneissic Rocks"
        },
        "colors": {
            1: (50, 205, 50, 255),     
            2: (255, 192, 203, 255),   
            3: (240, 240, 240, 255),   
            4: (128, 0, 128, 255),     
            5: (220, 20, 60, 255)      
        }
    },
    "Rainfall": {
        "classes": {1: "Very Low Rainfall", 2: "Low Rainfall", 3: "Moderate Rainfall", 4: "High Rainfall", 5: "Very High Rainfall"},
        "colors": {
            1: (255, 165, 0, 255), 
            2: (255, 255, 0, 255), 
            3: (0, 128, 0, 255), 
            4: (0, 255, 255, 255), 
            5: (0, 0, 139, 255)
        }
    },
    "Lineament Density": {
        "classes": {1: "Very Low Density", 2: "Low Density", 3: "Moderate Density", 4: "High Density", 5: "Very High Density"},
        "colors": {
            1: (34, 139, 34, 255), 
            2: (144, 238, 144, 255), 
            3: (255, 255, 0, 255), 
            4: (255, 140, 0, 255), 
            5: (255, 0, 0, 255)
        }
    },
    "Drainage Density": {
        "classes": {1: "Very Low Density", 2: "Low Density", 3: "Moderate Density", 4: "High Density", 5: "Very High Density"},
        "colors": {
            1: (0, 0, 139, 255), 
            2: (0, 191, 255, 255), 
            3: (255, 255, 0, 255), 
            4: (255, 165, 0, 255), 
            5: (255, 0, 0, 255)
        }
    },
    "Slope": {
        "classes": {1: "Very High", 2: "High", 3: "Moderate", 4: "Low", 5: "Very Low"},
        "colors": {
            1: (34, 139, 34, 255), 
            2: (152, 251, 152, 255), 
            3: (255, 255, 0, 255), 
            4: (255, 165, 0, 255), 
            5: (255, 0, 0, 255)
        }
    }
}

# ---------------- MAP FILE PATHS ----------------
from pathlib import Path

# 1. Anchor to the directory where app.py is located
BASE_DIR = Path(__file__).resolve().parent

# 2. Navigate up two levels to reach the repository root, then into GWPZM OUTPUT
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

# 3. tif1.tif is in the same directory as app.py
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

# =====================================================================
# 🔍 LIVE PATH DIAGNOSTICS CONTROL PANEL
# =====================================================================
missing_files = []
if not os.path.exists(gw_tif_path):
    missing_files.append(("Main Groundwater TIF", gw_tif_path))

for name, path in TIFF_LAYERS_REGISTRY.items():
    if not os.path.exists(path):
        missing_files.append((f"{name} Layer", path))

if missing_files:
    with st.expander("⚠️ SYSTEM DIAGNOSTICS: Missing File Detected", expanded=True):
        st.warning("The system cannot track down the following files on your hard drive. Please double-check their names or paths:")
        for label, path in missing_files:
            st.markdown(f"* **{label}**: Code is trying to read: `{path}`")

if not os.path.exists(gw_tif_path):
    st.stop()

# ---------------- SAFE READ RASTER ----------------
@st.cache_data
def load_raster_data(filepath):
    if not os.path.exists(filepath):
        return None, None, None, None, None, None
    try:
        with rasterio.open(filepath) as src:
            data = src.read(1)
            bounds = src.bounds
            raster_transform = src.transform  
            width = src.width
            height = src.height
            crs = src.crs.to_string() if src.crs else "EPSG:4326"
        return data, bounds, raster_transform, width, height, crs
    except Exception as e:
        return None, None, None, None, None, None

data, bounds, raster_transform, width, height, crs_string = load_raster_data(gw_tif_path)

ALL_LAYERS_CACHE = {}
for name, path in TIFF_LAYERS_REGISTRY.items():
    l_data, l_bounds, l_trans, l_w, l_h, l_crs = load_raster_data(path)
    if l_data is not None:
        ALL_LAYERS_CACHE[name] = {
            "data": l_data, "bounds": l_bounds, "transform": l_trans, "crs": l_crs, "path": path
        }

# ---------------- HIGH SPEED RASTER VECTORIZATION ----------------
@st.cache_data
def create_hover_polygons(filepath, classes_mapping, scale_factor=0.05):
    if not os.path.exists(filepath):
        return gpd.GeoDataFrame()
        
    with rasterio.open(filepath) as src:
        new_width = int(src.width * scale_factor)
        new_height = int(src.height * scale_factor)
        r_data = src.read(1, out_shape=(new_height, new_width), resampling=Resampling.nearest)
        transform = rasterio.transform.from_bounds(
            src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top, new_width, new_height
        )
        r_crs = src.crs

    valid_keys = list(classes_mapping.keys())
    if valid_keys:
        mask = np.isin(r_data, valid_keys)
    else:
        mask = r_data > 0
        
    shape_gen = shapes(r_data.astype(np.int32), mask=mask, transform=transform)
    results = [{'properties': {'raster_val': int(v)}, 'geometry': s} for s, v in shape_gen]
    
    if not results:
        gdf = gpd.GeoDataFrame(columns=['raster_val', 'Class_Name', 'geometry'], crs=r_crs)
        return gdf.to_crs(epsg=4326) if r_crs else gdf
        
    gdf = gpd.GeoDataFrame.from_features(results, crs=r_crs)
    gdf['Class_Name'] = gdf['raster_val'].map(classes_mapping).fillna("Unknown Class")
    
    if not gdf.crs and r_crs:
         gdf.set_crs(r_crs, inplace=True)
         
    gdf = gdf.to_crs(epsg=4326)
    gdf['geometry'] = gdf['geometry'].simplify(0.01, preserve_topology=False)
    
    return gdf

hover_gdf_gw = create_hover_polygons(gw_tif_path, LAYER_MAPPINGS["Groundwater Potential"]["classes"])

# ---------------- DYNAMIC AREA STATISTICS FUNCTION WITH AREA ----------------
@st.cache_data
def calculate_area_statistics(data_array, layer_name, layer_config, transform):
    unique, counts = np.unique(data_array, return_counts=True)
    count_dict = dict(zip(unique, counts))
    
    classes_config = layer_config.get("classes", {})
    colors_config = layer_config.get("colors", {})
    
    res_x = abs(transform[0])
    res_y = abs(transform[4])
    
    if res_x < 0.1:
        pixel_width_m = res_x * 111320 * np.cos(np.radians(17.5))
        pixel_height_m = res_y * 111000
        pixel_area_km2 = (pixel_width_m * pixel_height_m) / 1_000_000.0
    else:
        pixel_area_km2 = (res_x * res_y) / 1_000_000.0
    
    def rgb_to_hex(color_tuple):
        r, g, b = color_tuple[0], color_tuple[1], color_tuple[2]
        return '#{:02x}{:02x}{:02x}'.format(r, g, b)

    total_valid_pixels = sum(count_dict.get(k, 0) for k, name in classes_config.items() if name != "Unknown")
    
    stats = []
    for key, name in classes_config.items():
        if name == "Unknown":
            continue
            
        count = count_dict.get(key, 0)
        percentage = (count / total_valid_pixels * 100) if total_valid_pixels > 0 else 0
        area_km2 = count * pixel_area_km2
        
        raw_color = colors_config.get(key, (128, 128, 128))
        hex_color = rgb_to_hex(raw_color)
        
        stats.append({
            "Zone": name, 
            "Pixels": count, 
            "Percentage": percentage, 
            "Area (Sq.Km)": area_km2,
            "Color": hex_color
        })
        
    return pd.DataFrame(stats)

# ---------------- BOUNDS CONVERSION ----------------
transformer = Transformer.from_crs(crs_string, "EPSG:4326", always_xy=True)
lon1, lat1 = transformer.transform(bounds.left, bounds.bottom)
lon2, lat2 = transformer.transform(bounds.right, bounds.top)
center_lat, center_lon = (lat1 + lat2) / 2, (lon1 + lon2) / 2

# ---------------- OVERLAY GENERATION ----------------
@st.cache_data
def generate_overlay_image_fast(data_array, colors_config):
    if data_array is None:
        return None
        
    rgba = np.zeros((data_array.shape[0], data_array.shape[1], 4), dtype=np.uint8)
    
    for val, color in colors_config.items():
        rgba_color = color if len(color) == 4 else (*color, 255)
        rgba[data_array == val] = rgba_color
            
    img = Image.fromarray(rgba, 'RGBA')
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"

# ---------------- SIDEBAR STRUCTURE ----------------
details_placeholder = st.sidebar.empty()
st.sidebar.markdown("---")
st.sidebar.subheader("🗺️ Map Layer Toggles")
active_layers = {}
checked_layers_list = []

show_gw = st.sidebar.checkbox("Show Groundwater Potential Zones", value=True)
if show_gw:
    checked_layers_list.append("Groundwater Potential")
    active_layers["Groundwater Potential"] = {
        "url": generate_overlay_image_fast(data, LAYER_MAPPINGS["Groundwater Potential"]["colors"]),
        "bounds": bounds,
        "crs": crs_string,
        "hover": hover_gdf_gw
    }

for layer_name in TIFF_LAYERS_REGISTRY.keys():
    if st.sidebar.checkbox(f"Show {layer_name}"):
        l_info = ALL_LAYERS_CACHE.get(layer_name)
        if l_info is not None:
            checked_layers_list.append(layer_name)
            layer_mapping = LAYER_MAPPINGS.get(layer_name, {})
            layer_hover_gdf = create_hover_polygons(l_info["path"], layer_mapping.get("classes", {}))
            active_layers[layer_name] = {
                "url": generate_overlay_image_fast(l_info["data"], layer_mapping.get("colors", {})),
                "bounds": l_info["bounds"],
                "crs": l_info["crs"],
                "hover": layer_hover_gdf
            }
        else:
            st.sidebar.error(f"⚠️ {layer_name} file not found.")

st.sidebar.markdown("---")
global_opacity = st.sidebar.slider("Global Layer Opacity", min_value=0.0, max_value=1.0, value=0.8, step=0.1)

# ---------------- MAP MODULE ----------------
@st.fragment
def interactive_map_module(details_target, active_layers_dict, opacity_gw):
    # ... (Keep existing map initialization code here) ...
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", attr="Google Satellite")

    # ... (Keep existing layer and hover logic here) ...
    for layer_name, layer_info in active_layers_dict.items():
        if layer_info["url"] and layer_info["bounds"]:
            trans = Transformer.from_crs(layer_info["crs"], "EPSG:4326", always_xy=True)
            l1, b1 = trans.transform(layer_info["bounds"].left, layer_info["bounds"].bottom)
            l2, b2 = trans.transform(layer_info["bounds"].right, layer_info["bounds"].top)
            
            ImageOverlay(
                image=layer_info["url"], 
                bounds=[[b1, l1], [b2, l2]], 
                opacity=opacity_gw,
                name=layer_name
            ).add_to(m)
            
            if layer_info.get("hover") is not None and not layer_info["hover"].empty:
                folium.GeoJson(
                    layer_info["hover"], 
                    name=f"{layer_name} Hover",
                    style_function=lambda x: {'fillOpacity': 0.0, 'color': '#00000000', 'weight': 0}, 
                    tooltip=folium.GeoJsonTooltip(
                        fields=['Class_Name'], 
                        aliases=[f'📍 {layer_name}:'], 
                        style="background-color: white; padding: 6px; border-radius: 4px; font-weight: bold;"
                    )
                ).add_to(m)

    map_data = st_folium(m, width=1200, height=550, returned_objects=["last_clicked"], key="gw_portal_map")

    with details_target:
        with st.container():
            st.subheader("📍 Clicked Location Details")
            if map_data and map_data.get("last_clicked"):
                lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
                
                # --- NEW: STYLED BOX FOR LAT/LONG ---
                st.markdown(f"""
                    <div style="
                        background-color: #f0f2f6; 
                        padding: 15px; 
                        border-radius: 10px; 
                        border: 2px solid #31333F; 
                        display: flex; 
                        justify-content: space-between;
                        margin-bottom: 20px;">
                        <div>
                            <small style="color: #555;">Latitude</small>
                            <div style="font-size: 20px; font-weight: bold;">{lat:.5f}°</div>
                        </div>
                        <div>
                            <small style="color: #555;">Longitude</small>
                            <div style="font-size: 20px; font-weight: bold;">{lon:.5f}°</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # --- (Rest of your pixel value lookup code remains below) ---
                def get_pixel_value(t_data, t_trans, t_crs, target_lat, target_lon):
                    try:
                        transformer_back = Transformer.from_crs("EPSG:4326", t_crs, always_xy=True)
                        x, y = transformer_back.transform(target_lon, target_lat)
                        row, col = rasterio.transform.rowcol(t_trans, x, y)
                        if 0 <= row < t_data.shape[0] and 0 <= col < t_data.shape[1]:
                            return int(t_data[row, col])
                        return None
                    except:
                        return None

                for active_name in active_layers_dict.keys():
                    # ... (Keep your existing loop for values) ...
                    if active_name == "Groundwater Potential":
                        val = get_pixel_value(data, raster_transform, crs_string, lat, lon)
                    else:
                        info = ALL_LAYERS_CACHE.get(active_name)
                        val = get_pixel_value(info["data"], info["transform"], info["crs"], lat, lon)

                    if val is not None:
                        mapping = LAYER_MAPPINGS.get(active_name, {})
                        classes = mapping.get("classes", {})
                        colors = mapping.get("colors", {})
                        
                        if val in classes:
                            text = classes[val]
                            bg_rgba = colors.get(val, (128, 128, 128, 255))
                            bg_hex = '#{:02x}{:02x}{:02x}'.format(bg_rgba[0], bg_rgba[1], bg_rgba[2])
                            brightness = (bg_rgba[0]*299 + bg_rgba[1]*587 + bg_rgba[2]*114) / 1000
                            text_color = "black" if brightness > 125 else "white"
                            
                            st.markdown(f"**{active_name}:**")
                            st.markdown(f'<div style="background-color:{bg_hex}; padding:8px; border-radius:5px; text-align:center; color:{text_color}; margin-bottom:10px;"><b>{text}</b></div>', unsafe_allow_html=True)
            else:
                st.info("Click a point on the map to see its class value.")

# ---------------- DASHBOARD UI TABS ----------------
tab1, tab2, tab3 = st.tabs(["💧 Interactive Portal", "🖼️ Maps Gallery", "📄 Data Information"])

with tab1:
    st.markdown("#### ⚙️ Map Viewer")
    interactive_map_module(details_placeholder, active_layers, global_opacity)

    st.markdown("---")
    st.subheader("📊 Spatial Attribute Statistics")
    
    if checked_layers_list:
        selected_dashboard_layer = checked_layers_list[-1]
        
        if selected_dashboard_layer == "Groundwater Potential":
            active_stat_data = data
            active_transform = raster_transform
        else:
            active_stat_data = ALL_LAYERS_CACHE[selected_dashboard_layer]["data"]
            active_transform = ALL_LAYERS_CACHE[selected_dashboard_layer]["transform"]
            
        current_layer_config = LAYER_MAPPINGS.get(selected_dashboard_layer, {})
        df_stats = calculate_area_statistics(active_stat_data, selected_dashboard_layer, current_layer_config, active_transform)
        
        if not df_stats.empty:
            df_stats = df_stats[df_stats['Zone'] != 'Unknown']
        
        if not df_stats.empty and df_stats['Pixels'].sum() > 0:
            dash_col1, dash_col2 = st.columns([2, 1])

            with dash_col1:
                fig = px.pie(
                    df_stats, 
                    values='Percentage', 
                    names='Zone',
                    color='Zone',
                    color_discrete_map={row['Zone']: row['Color'] for _, row in df_stats.iterrows()},
                    hole=0.4
                )
                fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#000000', width=1)))
                fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with dash_col2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.write(f"**Calculated Area Coverage: {selected_dashboard_layer}**")
                display_df = df_stats[['Zone', 'Area (Sq.Km)', 'Percentage']].copy()
                st.dataframe(
                    display_df.style.format({
                        'Area (Sq.Km)': '{:.2f} km²',
                        'Percentage': '{:.2f}%'
                    }), 
                    use_container_width=True, 
                    hide_index=True
                )

with tab2:
    st.subheader("🖼️ Static Maps Inventory Gallery")
    st.markdown("Browse and review the static cartographic maps of Medchal-Malkajgiri District.")

    gallery_maps = list(MAPS_REGISTRY.items())
    
    for i in range(0, len(gallery_maps), 2):
        col1, col2 = st.columns(2)
        
        with col1:
            name1, path1 = gallery_maps[i]
            if os.path.exists(path1):
                st.image(path1, caption=name1, use_container_width=True)
            else:
                st.error(f"⚠️ Image file missing: `{name1}`")
                
        with col2:
            if i + 1 < len(gallery_maps):
                name2, path2 = gallery_maps[i+1]
                if os.path.exists(path2):
                    st.image(path2, caption=name2, use_container_width=True)
                else: st.error(f"⚠️ Image file missing: `{name2}`")
# ... (Previous code for tab2) ...

with tab3:
    st.header("📄 Data Sources & Methodology")
    st.markdown("""
    This WebGIS portal maps Groundwater Potential Zones using a Multi-Criteria Decision Making (MCDM) approach. 
    The primary datasets were sourced based on their spatial resolution, temporal relevance, and scientific authenticity.

    ### 🗂️ Thematic Layers & Sources
    | Dataset | Source | Type / Resolution | Primary Purpose |
    | :--- | :--- | :--- | :--- |
    | **DEM** | Bhoonidhi Geoportal | 30m Raster | Terrain, slope, and drainage analysis |
    | **LULC** | Bhoonidhi (LISS-IV) | Vector/Raster | Surface runoff and recharge evaluation |
    | **Rainfall** | IMD | Annual Data | Groundwater recharge estimation |
    | **Soil** | SoilGrids | Raster | Soil texture and infiltration capacity |
    | **Geology** | NGDR | Vector | Lithological and aquifer characterization |
    | **Geomorphology** | NGDR | Vector | Landform and structural analysis |
    | **Lineaments** | Bhuvan WMS | Vector | Fault and fracture density mapping |
    | **Drainage** | Derived from DEM | Vector | Runoff and stream network density |
    
    ### ⚙️ Analytical Processing (AHP)
    The individual thematic layers were processed in a GIS environment. We utilized **Saaty’s Analytical Hierarchy Process (AHP)** to calculate the normalized weights for each layer based on their relative importance to groundwater occurrence. 
    
    The final Groundwater Potential Index (GWPI) was calculated using a Weighted Overlay Analysis, classifying the region into five distinct zones ranging from **Very Low** to **Very High**.
    """)
    
