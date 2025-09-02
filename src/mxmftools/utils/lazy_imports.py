import importlib
import typing as _t


def lazy_load_module_symbols(
    symbols: dict[str, str], module_globals: dict[_t.Any, _t.Any]
):
    """
    给模块自动添加懒加载支持。

    symbols: {符号名: 模块路径}，模块路径可以是相对或绝对路径
    module_globals: 模块的 globals()
    """
    # 更新 __all__
    module_globals.setdefault("__all__", [])
    module_globals["__all__"].extend(symbols.keys())

    _lazy_imports = symbols.copy()

    def __getattr__(name: str) -> _t.Any:
        if name in _lazy_imports:
            module_name = _lazy_imports[name]
            module = importlib.import_module(
                module_name, package=module_globals.get("__name__")
            )
            attr = getattr(module, name)
            module_globals[name] = attr  # 缓存
            return attr
        raise AttributeError(
            f"module {module_globals.get('__name__')!r} has no attribute {name!r}"
        )

    def __dir__():
        # dir(module) 显示懒加载符号
        return list(module_globals.keys()) + list(_lazy_imports.keys())

    module_globals["__getattr__"] = __getattr__
    module_globals["__dir__"] = __dir__
