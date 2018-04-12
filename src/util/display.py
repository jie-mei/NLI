import inspect
from typing import List

from sklearn import base as skb


class ReprMixin:
    """ Mixin for formating `__repr__`. """

    @classmethod
    def _get_hyperparam_names(cls) -> List[str]:
        if cls.__init__ == object.__init__:
            # No explicit constructor to introspect
            return []
        return sorted(list(inspect.signature(cls.__init__).parameters))

    def get_hyperparams(self) -> dict:
        """Get parameters of this object.

        All hyperparameters must be initialized from the constructor and their
        values should be stored as model properties.

        Returns:
            Mapping of parameter name string mapped to their values.
        """
        return {n: getattr(self, n, None) for n in self._get_hyperparam_names()
                if n not in ['self', 'args', 'kwargs'] and n in self.__dict__}

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        param_list = skb._pprint(self.get_hyperparams(), offset=14)  # 8 chars
        return '%s(%s)' % (class_name, param_list)
