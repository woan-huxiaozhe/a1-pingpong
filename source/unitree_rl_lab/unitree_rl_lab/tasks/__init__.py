##
# Register Gym environments.
##

try:
    from isaaclab_tasks.utils import import_packages
except ModuleNotFoundError:
    # Allows importing Isaac-independent utilities (for example SAC replay tests)
    # from a plain Python environment. Isaac task registration still runs inside
    # the Isaac Lab launcher where isaaclab_tasks is available.
    import_packages = None

if import_packages is not None:
    # The blacklist is used to prevent importing configs from sub-packages
    _BLACKLIST_PKGS = []
    # Import all configs in this package
    import_packages(__name__, _BLACKLIST_PKGS)
