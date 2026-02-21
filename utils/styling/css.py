"""
CSS Styling Module
==================
CSS styles for Streamlit UI components.
"""

# Initial page input styling
initial_page = """
<style>
    div[data-baseweb="input"] {
        background-color: #d3d3d3;
        border-radius: 8px;
        padding: 5px;
    }

    input {
        background-color: #d3d3d3 !important;
        color: black !important;
        border: none !important;
    }

    button[aria-label="decrement"], 
    button[aria-label="increment"] {
        background-color: #32CD32 !important;
        color: white !important;
        border-radius: 5px;
        border: none !important;
        width: 30px;
        height: 30px;
    }
</style>
"""

# Select box styling
select_box = """
<style>
    div[data-baseweb="select"] {
        background-color: #D3D3D3 !important;
        border-radius: 5px !important;
        padding: 5px !important;
        margin: 0px;
    }
</style>
"""

# Responsive table styling
responsive_table_styling = """
<style>
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th {
        text-align: center;
        padding: 8px;
        border: 1px solid #ddd;
        background-color: #ff9900;
        color: white;
    }
    td {
        text-align: left;
        padding: 8px;
        border: 1px solid #ddd;
    }
    tr:nth-child(even) {
        background-color: #f2f2f2;
    }
</style>
"""

# Page navigation button styling
choose_page_option = """
<style>
    .element-container:has(style) {
        display: none;
    }
    #button-after {
        display: none;
    }
    .element-container:has(#button-after) {
        display: none;
    }
    .element-container:has(#button-after) + div button {
        background-color: white;
        width: 300px;
        height: 50px;
        font-size: 500px;
        border-radius: 8px;
        color: black;
        border: 2px solid red;
    }
    .button-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
    }
</style>
"""
