from typing import TypeVar
import optika

__all__ = [
    "SagT",
    "MaterialT",
    "RulingT",
]


#: Generic sag type.
#: Should be :obj:`None` or an instance of
#: :class:`optika.sags.AbstractSag`
SagT = TypeVar("SagT", bound=None | optika.sags.AbstractSag)

#: Generic material type.
#: Should be :obj:`None` or an instance of
#: :class:`optika.materials.AbstractMaterial`
MaterialT = TypeVar("MaterialT", bound=None | optika.materials.AbstractMaterial)

#: Generic ruling type.
#: Should be :obj:`None` or an instance of
#: :class:`optika.rulings.AbstractRulings`
RulingT = TypeVar("RulingT", bound=None | optika.rulings.AbstractRulings)
