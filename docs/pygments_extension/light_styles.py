from pygments.token import Name
from pygments.styles.emacs import EmacsStyle
from pygments.styles._mapping import STYLES
from pygments.styles import _STYLE_NAME_TO_MODULE_MAP


glacier = "#1f6feb"


class XTunerEmacsStyle(EmacsStyle):
    styles = EmacsStyle.styles.copy()
    styles[Name.Keyword] = glacier


_STYLE_NAME_TO_MODULE_MAP["xtuner-emacs"] = ("docs.pygments_extension.light_styles", "XTunerEmacsStyle")
STYLES["XTunerEmacsStyle"] = ("docs.pygments_extension.light_styles", "xtuner-emacs", ())
