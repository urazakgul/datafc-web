def main_page_header():
    return """
    <h1 style="text-align:center; font-size:2.5em;">
        <span style="font-weight:bold; color: #fff;">DATAFC-WEB</span>
        <span style="font-weight:normal; color:#888;"> [Football Analytics App]</span>
    </h1>
    """

def support_button():
    return """
    <style>
    .support-btn:hover {
        background-color: #ffeb77 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.18) !important;
    }
    </style>
    <div style="display: flex; justify-content: center; align-items: center; margin-top: 24px; margin-bottom: 18px;">
        <a href="https://www.buymeacoffee.com/urazdev" target="_blank" class="support-btn" style="
            display: inline-block;
            background-color: #FFDD00;
            color: #333333;
            font-weight: bold;
            border-radius: 8px;
            padding: 6px 18px;
            font-size: 15px;
            text-decoration: none;
            box-shadow: 0 1px 4px rgba(0,0,0,0.10);
            border: 1px solid #ffe066;
            transition: background 0.2s, box-shadow 0.2s;
            text-align: center;
            cursor: pointer;">
            ☕ Click here to support this project
        </a>
    </div>
    """

def page_styles():
    return """
    <style>
    .main {
        padding-top: 0rem !important;
    }
    .block-container {
        width: 60vw !important;
        max-width: 1400px !important;
        min-width: 350px;
        margin: auto;
        margin-top: 0px !important;
        padding-left: 24px;
        padding-right: 24px;
    }
    </style>
    """

def team_analysis_header():
    return """
    <h3 style="text-align:center; font-weight:normal;">Team Analysis</h3>
    """

def prediction_header():
    return """
    <h3 style="text-align:center; font-weight:normal;">Prediction</h3>
    """