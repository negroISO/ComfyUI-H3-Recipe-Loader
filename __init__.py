from .h3_recipe_loader import (NODE_CLASS_MAPPINGS,
                               NODE_DISPLAY_NAME_MAPPINGS)

# Serves web/h3_recipe_loader.js, which live-fills the settings widgets when
# the checkpoint changes. Without this the nodes still work; they just stop
# auto-updating.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
           "WEB_DIRECTORY"]
