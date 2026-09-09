import importlib
import pkgutil


def exported_symbols(db_module, **opts):
    symbols = {}
    if "__exports__" in db_module.__dict__:
        symbols.update(
            {export: db_module.__dict__[export] for export in db_module.__exports__}
        )
    if "__opt_exports__" in db_module.__dict__:
        symbols.update(db_module.__opt_exports__(opts))
    return symbols


def import_exports(package, recursive=True, **opts):
    """Returns a dictionary of exported items from a database"""
    results = {}
    if isinstance(package, str):
        package = importlib.import_module(package)
        results.update(exported_symbols(package, **opts))
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + "." + name
        db_module = importlib.import_module(full_name)
        results.update(exported_symbols(db_module, **opts))
        if recursive and is_pkg:
            results.update(import_exports(full_name))
    return results


class Database:
    def __init__(self, package, **kwargs):
        self.__dict__.update(import_exports(package, **kwargs))

    def __getitem__(self, key):
        if key not in self.__dict__:
            raise RuntimeError("Key not in database: %s" % key)
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __contains__(self, key):
        return key in self.__dict__

    def load_package(self, package, **kwargs):
        self.__dict__.update(import_exports(package, **kwargs))
