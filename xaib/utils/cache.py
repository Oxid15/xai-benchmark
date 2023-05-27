import os
from typing import Any
from ..base import Factory, Model


class ModelCache:
    def __init__(self, factory: Factory, root='./.xaib/models') -> None:
        self._factory = factory
        self._root = root

        if not os.path.exists(self._root):
            os.makedirs(self._root, exist_ok=True)

        self._keys = os.listdir(self._root)

    def get(self, name: str, *args: Any, key: str = "", **kwargs: Any) -> Model:
        full_name = name
        if key:
            full_name = '_'.join((name, key))

        if full_name in self._keys:
            clss = self._factory.get_constructor(name)
            model = clss(name=name)
            model.load(
                os.path.join(self._root, full_name, "model")
            )  # The use of filename should be deprecated
            return model

        model = self._factory.get(name, *args, **kwargs)

        os.makedirs(os.path.join(self._root, full_name), exist_ok=True)
        model.save(os.path.join(self._root, full_name, "model"))
        self._keys.append(full_name)
        return model

    def add(self, *args: Any, **kwargs: Any):
        '''
        See Factory.add method
        '''
        self._factory.add(*args, **kwargs)
