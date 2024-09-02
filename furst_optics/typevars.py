from typing import TypeVar
import optika

__all__ = [
    "SagT",
    "MaterialT",
    "RulingT",
]


#: Generic sag type
SagT = TypeVar("SagT", bound=None | optika.sags.AbstractSag)

#: Generic material type
MaterialT = TypeVar("MaterialT", bound=None | optika.materials.AbstractMaterial)

#: Generic ruling type
RulingT = TypeVar("RulingT", bound=None | optika.rulings.AbstractRulings)