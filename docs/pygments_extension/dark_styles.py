from pygments.token import Name
from pygments.styles.dracula import DraculaStyle
from pygments.styles._mapping import STYLES
from pygments.styles import _STYLE_NAME_TO_MODULE_MAP

from pygments import unistring as uni

glacier = "#55bfe3"


class XTunerDraculaStyle(DraculaStyle):
    styles = DraculaStyle.styles.copy()
    styles[Name.Keyword] = glacier


_STYLE_NAME_TO_MODULE_MAP["xtuner-dracula"] = ("docs.pygments_extension.dark_styles", "XTunerDraculaStyle")
STYLES["XTunerDraculaStyle"] = ("docs.pygments_extension.dark_styles", "xtuner-dracula", ())
