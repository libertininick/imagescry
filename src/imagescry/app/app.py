"""Image annotation application."""

import json

import dash
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

from imagescry.image.info import ImageShape
from imagescry.image.io import read_image_and_encode

# Dash web application
web_app = dash.Dash(__name__)

# Path to your image file
IMAGE_PATH = "data/ds/RGB/S1026398.JPG"
MAX_SIZE = 1_280

# Get image info and scale it down
encoded_image = read_image_and_encode(IMAGE_PATH)
original_height, original_width = ImageShape.from_source(IMAGE_PATH)

# Scale down the image for display (adjust MAX_SIZE as needed)
scale_factor = min(MAX_SIZE / original_width, MAX_SIZE / original_height, 1.0)
img_width = int(original_width * scale_factor)
img_height = int(original_height * scale_factor)

web_app.layout = html.Div([
    html.H1("Image Polygon Annotator", style={"textAlign": "center"}),
    # Image display using Plotly
    dcc.Graph(id="image-graph", config={"modeBarButtonsToAdd": ["drawclosedpath"], "displayModeBar": True}),
    # Workflow buttons for clearing and saving polygons
    html.Div(
        [
            html.Button(
                "Clear All Polygons",
                id="clear-btn",
                n_clicks=0,
                style={"margin": "10px", "padding": "10px", "backgroundColor": "#ff4444", "color": "white"},
            ),
            html.Button(
                "Save Polygons",
                id="save-btn",
                n_clicks=0,
                style={"margin": "10px", "padding": "10px", "backgroundColor": "#44ff44", "color": "black"},
            ),
        ],
        style={"textAlign": "center"},
    ),
    html.Div(id="save-status", style={"textAlign": "center", "margin": "20px"}),
    # Store component to hold polygon data
    dcc.Store(id="polygon-store", data=[]),
])


@web_app.callback(Output("image-graph", "figure"), [Input("clear-btn", "n_clicks")], [State("polygon-store", "data")])
def update_graph(clear_clicks, stored_polygons) -> go.Figure:
    """Update the graph with image and any existing polygons."""
    fig = go.Figure()

    # Add the image
    if encoded_image:
        fig.add_layout_image({
            "source": encoded_image,
            "xref": "x",
            "yref": "y",
            "x": 0,
            "y": img_height,
            "sizex": img_width,
            "sizey": img_height,
            "sizing": "stretch",
            "opacity": 1,
            "layer": "below",
        })

    # Set up the layout
    fig.update_layout(
        width=img_width,
        height=img_height + 100,
        xaxis={"range": [0, img_width], "showgrid": False, "zeroline": False, "showticklabels": False},
        yaxis={
            "range": [0, img_height],
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        plot_bgcolor="white",
        margin={"l": 0, "r": 0, "t": 50, "b": 50},
        dragmode="drawclosedpath",
        newshape={"line_color": "red", "line_width": 3, "fillcolor": "rgba(255, 0, 0, 0.3)"},
    )

    return fig


@web_app.callback(
    [Output("polygon-store", "data"), Output("save-status", "children")],
    [Input("image-graph", "relayoutData"), Input("save-btn", "n_clicks")],
    [State("polygon-store", "data")],
)
def handle_drawing_and_saving(relayoutData, save_clicks, stored_polygons):
    """Handle polygon drawing and saving."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return stored_polygons, ""

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger_id == "image-graph" and relayoutData:
        polygons = process_drawn_shapes(relayoutData)
        return polygons, f"Current polygons: {len(polygons)}"

    elif trigger_id == "save-btn" and save_clicks > 0:
        status_message = save_polygons_to_file(stored_polygons)
        return stored_polygons, status_message

    return stored_polygons, ""


def parse_svg_path(path_string):
    """Parse SVG path string to extract polygon coordinates."""
    try:
        # Simple parser for closed paths created by Plotly
        # Format: "M x1,y1 L x2,y2 L x3,y3 ... Z"
        coords = []

        # Remove 'M' and 'Z', split by 'L'
        path_clean = path_string.replace("M", "").replace("Z", "").strip()
        points = path_clean.split("L")

        for point in points:
            if point.strip():
                try:
                    x, y = point.strip().split(",")
                    coords.append([float(x), float(y)])
                except:
                    continue

        return coords if len(coords) >= 3 else None
    except:
        return None


def get_zoom_pan_info(relayoutData):
    """Extract zoom and pan information from relayoutData."""
    x_range = relayoutData.get("xaxis.range", [0, img_width])
    y_range = relayoutData.get("yaxis.range", [0, img_height])

    x_zoom_factor = img_width / (x_range[1] - x_range[0])
    y_zoom_factor = img_height / (y_range[1] - y_range[0])

    return {
        "x_range": x_range,
        "y_range": y_range,
        "x_zoom_factor": x_zoom_factor,
        "y_zoom_factor": y_zoom_factor,
        "x_offset": x_range[0],
        "y_offset": y_range[0],
    }


def transform_coordinates_to_original(coords, zoom_info):
    """Transform display coordinates to original image coordinates."""
    original_coords = []

    for x, y in coords:
        # Account for zoom and pan
        actual_x = (x + zoom_info["x_offset"]) / zoom_info["x_zoom_factor"]
        actual_y = (y + zoom_info["y_offset"]) / zoom_info["y_zoom_factor"]

        # Scale back to original image size
        orig_x = actual_x / scale_factor
        orig_y = actual_y / scale_factor

        # Convert y from cartesian to image coordinates
        orig_y = original_height - orig_y

        original_coords.append([orig_x, orig_y])

    return original_coords


def create_polygon_data(shape, zoom_info, polygon_index):
    """Create a polygon data structure from a shape."""
    if shape["type"] != "path":
        return None

    coords = parse_svg_path(shape["path"])
    if not coords:
        return None

    original_coords = transform_coordinates_to_original(coords, zoom_info)

    return {
        "coordinates": original_coords,
        "original_image_size": [original_width, original_height],
        "display_scale_factor": scale_factor,
        "zoom_info": {
            "x_range": zoom_info["x_range"],
            "y_range": zoom_info["y_range"],
            "x_zoom_factor": zoom_info["x_zoom_factor"],
            "y_zoom_factor": zoom_info["y_zoom_factor"],
        },
        "label": f"Polygon_{polygon_index}",
    }


def process_drawn_shapes(relayoutData):
    """Process newly drawn shapes and convert to polygon data."""
    if "shapes" not in relayoutData:
        return []

    zoom_info = get_zoom_pan_info(relayoutData)
    polygons = []

    for i, shape in enumerate(relayoutData["shapes"], 1):
        polygon_data = create_polygon_data(shape, zoom_info, i)
        if polygon_data:
            polygons.append(polygon_data)

    return polygons


def save_polygons_to_file(polygons):
    """Save polygons to JSON file and return status message."""
    if not polygons:
        return "⚠️ No polygons to save"

    try:
        with open("polygons.json", "w", encoding="utf-8") as f:
            json.dump(polygons, f, indent=2)
        return f"✅ Saved {len(polygons)} polygons to polygons.json"
    except Exception as e:
        return f"❌ Error saving: {e!s}"


if __name__ == "__main__":
    web_app.run(debug=True)
